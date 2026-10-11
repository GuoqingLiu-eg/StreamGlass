#include "streamglass/bluez_media_controller.hpp"

#include "streamglass/pause_control.hpp"

#include <gio/gio.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <iostream>
#include <initializer_list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace streamglass {

namespace {

constexpr const char* kBluezService = "org.bluez";
constexpr const char* kPlayerPath = "/org/streamglass/player";
constexpr const char* kPlayerInterface = "org.mpris.MediaPlayer2.Player";
constexpr const char* kPropertiesInterface =
    "org.freedesktop.DBus.Properties";
constexpr const char* kAvrcpControllerUuid =
    "0000110e-0000-1000-8000-00805f9b34fb";

constexpr const char* kPlayerXml = R"xml(
<node>
  <interface name="org.mpris.MediaPlayer2.Player">
    <method name="Play"/>
    <method name="Pause"/>
    <method name="PlayPause"/>
    <method name="Stop"/>
    <method name="Next"/>
    <method name="Previous"/>
    <method name="Seek">
      <arg name="Offset" type="x" direction="in"/>
    </method>
    <method name="SetPosition">
      <arg name="TrackId" type="o" direction="in"/>
      <arg name="Position" type="x" direction="in"/>
    </method>
    <method name="OpenUri">
      <arg name="Uri" type="s" direction="in"/>
    </method>
    <property name="PlaybackStatus" type="s" access="read"/>
    <property name="LoopStatus" type="s" access="readwrite"/>
    <property name="Rate" type="d" access="readwrite"/>
    <property name="Shuffle" type="b" access="readwrite"/>
    <property name="Metadata" type="a{sv}" access="read"/>
    <property name="Volume" type="d" access="readwrite"/>
    <property name="Position" type="x" access="read"/>
    <property name="MinimumRate" type="d" access="read"/>
    <property name="MaximumRate" type="d" access="read"/>
    <property name="CanGoNext" type="b" access="read"/>
    <property name="CanGoPrevious" type="b" access="read"/>
    <property name="CanPlay" type="b" access="read"/>
    <property name="CanPause" type="b" access="read"/>
    <property name="CanSeek" type="b" access="read"/>
    <property name="CanControl" type="b" access="read"/>
  </interface>
</node>
)xml";

std::string uppercase(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](const unsigned char character) {
            return static_cast<char>(std::toupper(character));
        });
    return value;
}

std::string find_adapter_path(
    GDBusConnection* connection,
    const std::string& controller_address) {
    const auto expected = uppercase(controller_address);
    GError* error = nullptr;
    GVariant* reply = g_dbus_connection_call_sync(
        connection, kBluezService, "/",
        "org.freedesktop.DBus.ObjectManager", "GetManagedObjects",
        nullptr, G_VARIANT_TYPE("(a{oa{sa{sv}}})"),
        G_DBUS_CALL_FLAGS_NONE, 3000, nullptr, &error);
    if (reply == nullptr) {
        if (error != nullptr) {
            std::cerr << "[control] BlueZ object discovery failed: "
                      << error->message << "\n";
        }
        g_clear_error(&error);
        return {};
    }

    std::string result;
    GVariant* objects = g_variant_get_child_value(reply, 0);
    GVariantIter iterator;
    g_variant_iter_init(&iterator, objects);
    const gchar* object_path = nullptr;
    GVariant* interfaces = nullptr;
    while (g_variant_iter_next(
        &iterator, "{&o@a{sa{sv}}}", &object_path, &interfaces)) {
        GVariant* properties = g_variant_lookup_value(
            interfaces, "org.bluez.Adapter1",
            G_VARIANT_TYPE("a{sv}"));
        if (properties != nullptr) {
            const gchar* address = nullptr;
            if (g_variant_lookup(
                    properties, "Address", "&s", &address) &&
                address != nullptr &&
                uppercase(address) == expected) {
                result = object_path;
            }
            g_variant_unref(properties);
        }
        g_variant_unref(interfaces);
        if (!result.empty()) {
            break;
        }
    }
    g_variant_unref(objects);
    g_variant_unref(reply);
    return result;
}

std::string device_path(
    const std::string& adapter_path,
    const std::string& device_address) {
    auto token = uppercase(device_address);
    std::replace(token.begin(), token.end(), ':', '_');
    return adapter_path + "/dev_" + token;
}

bool is_expected_bluez_error(
    GError* error,
    const std::initializer_list<const char*> names) {
    if (error == nullptr || !g_dbus_error_is_remote_error(error)) {
        return false;
    }
    gchar* remote = g_dbus_error_get_remote_error(error);
    if (remote == nullptr) {
        return false;
    }
    const std::string value(remote);
    g_free(remote);
    return std::any_of(
        names.begin(), names.end(),
        [&](const char* name) { return value == name; });
}

}  // namespace

class BluezMediaController::Impl final {
public:
    Impl(
        PauseControl& pause_control,
        std::string controller_address,
        std::string device_address)
        : pause_control_(pause_control),
          controller_address_(std::move(controller_address)),
          device_address_(std::move(device_address)) {}

    ~Impl() { stop(); }

    void start() {
        if (worker_.joinable()) {
            return;
        }
        stop_requested_.store(false, std::memory_order_release);
        worker_ = std::thread([this] { run(); });
    }

    void stop() noexcept {
        stop_requested_.store(true, std::memory_order_release);
        {
            const std::scoped_lock lock(context_mutex_);
            if (context_ != nullptr) {
                g_main_context_invoke(
                    context_,
                    [](gpointer data) -> gboolean {
                        auto* self = static_cast<Impl*>(data);
                        if (self->loop_ != nullptr) {
                            g_main_loop_quit(self->loop_);
                        }
                        return G_SOURCE_REMOVE;
                    },
                    this);
            }
        }
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    static void handle_method_call(
        GDBusConnection* connection,
        const gchar*,
        const gchar*,
        const gchar*,
        const gchar* method_name,
        GVariant*,
        GDBusMethodInvocation* invocation,
        gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        bool state_changed = false;
        if (std::string_view(method_name) == "Play") {
            state_changed = self->pause_control_.set_paused(false);
        } else if (std::string_view(method_name) == "Pause" ||
                   std::string_view(method_name) == "Stop") {
            state_changed = self->pause_control_.set_paused(true);
        } else if (std::string_view(method_name) == "PlayPause") {
            static_cast<void>(self->pause_control_.toggle());
            state_changed = true;
        }

        if (state_changed) {
            self->emit_playback_status(connection);
            std::cerr << "[control] AVRCP " << method_name
                      << " requested "
                      << (self->pause_control_.paused() ? "pause" : "resume")
                      << "\n";
        }
        g_dbus_method_invocation_return_value(invocation, nullptr);
    }

    static GVariant* get_property(
        GDBusConnection*,
        const gchar*,
        const gchar*,
        const gchar*,
        const gchar* property_name,
        GError**,
        gpointer user_data) {
        const auto* self = static_cast<Impl*>(user_data);
        const std::string_view name(property_name);
        if (name == "PlaybackStatus") {
            return g_variant_new_string(
                self->pause_control_.paused() ? "Paused" : "Playing");
        }
        if (name == "LoopStatus") {
            return g_variant_new_string("None");
        }
        if (name == "Rate" || name == "Volume" ||
            name == "MinimumRate" || name == "MaximumRate") {
            return g_variant_new_double(1.0);
        }
        if (name == "Shuffle" || name == "CanGoNext" ||
            name == "CanGoPrevious" || name == "CanSeek") {
            return g_variant_new_boolean(false);
        }
        if (name == "CanPlay" || name == "CanPause" ||
            name == "CanControl") {
            return g_variant_new_boolean(true);
        }
        if (name == "Position") {
            return g_variant_new_int64(0);
        }
        if (name == "Metadata") {
            GVariantBuilder metadata;
            g_variant_builder_init(
                &metadata, G_VARIANT_TYPE("a{sv}"));
            g_variant_builder_add(
                &metadata, "{sv}", "mpris:trackid",
                g_variant_new_object_path(
                    "/org/streamglass/player/translation"));
            g_variant_builder_add(
                &metadata, "{sv}", "xesam:title",
                g_variant_new_string("StreamGlass Translation"));
            return g_variant_builder_end(&metadata);
        }
        return nullptr;
    }

    static gboolean set_property(
        GDBusConnection*,
        const gchar*,
        const gchar*,
        const gchar*,
        const gchar*,
        GVariant*,
        GError**,
        gpointer) {
        return true;
    }

    static gboolean retry_registration(gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        if (self->stop_requested_.load(std::memory_order_acquire)) {
            return G_SOURCE_REMOVE;
        }
        if (!self->registered_) {
            self->register_with_bluez();
        }
        return G_SOURCE_CONTINUE;
    }

    static void bluez_owner_changed(
        GDBusConnection*,
        const gchar*,
        const gchar*,
        const gchar*,
        const gchar*,
        GVariant* parameters,
        gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        const gchar* name = nullptr;
        const gchar* old_owner = nullptr;
        const gchar* new_owner = nullptr;
        g_variant_get(
            parameters, "(&s&s&s)", &name, &old_owner, &new_owner);
        if (std::string_view(name) != kBluezService) {
            return;
        }
        self->registered_ = false;
        if (new_owner != nullptr && new_owner[0] != '\0') {
            self->register_with_bluez();
        }
    }

    void emit_playback_status(GDBusConnection* connection) const {
        GVariantBuilder changed;
        g_variant_builder_init(&changed, G_VARIANT_TYPE("a{sv}"));
        g_variant_builder_add(
            &changed, "{sv}", "PlaybackStatus",
            g_variant_new_string(
                pause_control_.paused() ? "Paused" : "Playing"));

        GVariantBuilder invalidated;
        g_variant_builder_init(&invalidated, G_VARIANT_TYPE("as"));
        GError* error = nullptr;
        g_dbus_connection_emit_signal(
            connection, nullptr, kPlayerPath, kPropertiesInterface,
            "PropertiesChanged",
            g_variant_new(
                "(s@a{sv}@as)", kPlayerInterface,
                g_variant_builder_end(&changed),
                g_variant_builder_end(&invalidated)),
            &error);
        if (error != nullptr) {
            std::cerr << "[control] property signal failed: "
                      << error->message << "\n";
            g_error_free(error);
        }
    }

    void register_with_bluez() {
        adapter_path_ =
            find_adapter_path(connection_, controller_address_);
        if (adapter_path_.empty()) {
            if (!adapter_missing_logged_) {
                std::cerr << "[control] Bluetooth adapter "
                          << controller_address_
                          << " is not available; retrying\n";
                adapter_missing_logged_ = true;
            }
            return;
        }
        adapter_missing_logged_ = false;

        GVariantBuilder options;
        g_variant_builder_init(&options, G_VARIANT_TYPE("a{sv}"));
        g_variant_builder_add(
            &options, "{sv}", "PlaybackStatus",
            g_variant_new_string(
                pause_control_.paused() ? "paused" : "playing"));
        g_variant_builder_add(
            &options, "{sv}", "CanPlay", g_variant_new_boolean(true));
        g_variant_builder_add(
            &options, "{sv}", "CanPause", g_variant_new_boolean(true));
        g_variant_builder_add(
            &options, "{sv}", "CanControl", g_variant_new_boolean(true));
        g_variant_builder_add(
            &options, "{sv}", "Identity",
            g_variant_new_string("StreamGlass"));

        GError* error = nullptr;
        GVariant* reply = g_dbus_connection_call_sync(
            connection_, kBluezService, adapter_path_.c_str(),
            "org.bluez.Media1", "RegisterPlayer",
            g_variant_new(
                "(o@a{sv})", kPlayerPath,
                g_variant_builder_end(&options)),
            nullptr, G_DBUS_CALL_FLAGS_NONE, 3000, nullptr, &error);
        if (reply == nullptr) {
            if (is_expected_bluez_error(
                    error, {"org.bluez.Error.AlreadyExists"})) {
                registered_ = true;
            } else {
                if (error != nullptr) {
                    std::cerr << "[control] BlueZ registration retry: "
                              << error->message << "\n";
                }
                g_clear_error(&error);
                return;
            }
        } else {
            g_variant_unref(reply);
            registered_ = true;
        }
        g_clear_error(&error);

        const auto remote_path =
            device_path(adapter_path_, device_address_);
        reply = g_dbus_connection_call_sync(
            connection_, kBluezService, remote_path.c_str(),
            "org.bluez.Device1", "ConnectProfile",
            g_variant_new("(s)", kAvrcpControllerUuid), nullptr,
            G_DBUS_CALL_FLAGS_NONE, 3000, nullptr, &error);
        if (reply != nullptr) {
            g_variant_unref(reply);
        } else if (!is_expected_bluez_error(
                       error,
                       {"org.bluez.Error.AlreadyConnected",
                        "org.bluez.Error.InProgress",
                        "org.bluez.Error.NotAvailable"})) {
            if (error != nullptr) {
                std::cerr << "[control] AVRCP profile connection: "
                          << error->message << "\n";
            }
        }
        g_clear_error(&error);
        std::cerr << "[control] BlueZ media player registered adapter="
                  << adapter_path_ << "\n";
    }

    void unregister_from_bluez() noexcept {
        if (!registered_ || connection_ == nullptr ||
            adapter_path_.empty()) {
            return;
        }
        GError* error = nullptr;
        GVariant* reply = g_dbus_connection_call_sync(
            connection_, kBluezService, adapter_path_.c_str(),
            "org.bluez.Media1", "UnregisterPlayer",
            g_variant_new("(o)", kPlayerPath), nullptr,
            G_DBUS_CALL_FLAGS_NONE, 1500, nullptr, &error);
        if (reply != nullptr) {
            g_variant_unref(reply);
        }
        g_clear_error(&error);
        registered_ = false;
    }

    void run() noexcept {
        GMainContext* context = g_main_context_new();
        GMainLoop* loop = g_main_loop_new(context, false);
        {
            const std::scoped_lock lock(context_mutex_);
            context_ = context;
            loop_ = loop;
        }
        g_main_context_push_thread_default(context);

        GError* error = nullptr;
        connection_ = g_bus_get_sync(G_BUS_TYPE_SYSTEM, nullptr, &error);
        if (connection_ == nullptr) {
            std::cerr << "[control] system D-Bus unavailable: "
                      << (error == nullptr ? "unknown" : error->message)
                      << "\n";
            g_clear_error(&error);
        } else {
            introspection_ =
                g_dbus_node_info_new_for_xml(kPlayerXml, &error);
            if (introspection_ == nullptr) {
                std::cerr << "[control] MPRIS introspection failed: "
                          << (error == nullptr ? "unknown" : error->message)
                          << "\n";
                g_clear_error(&error);
            } else {
                static const GDBusInterfaceVTable vtable{
                    .method_call = handle_method_call,
                    .get_property = get_property,
                    .set_property = set_property,
                    .padding = {nullptr},
                };
                object_id_ = g_dbus_connection_register_object(
                    connection_, kPlayerPath,
                    introspection_->interfaces[0], &vtable, this,
                    nullptr, &error);
                if (object_id_ == 0) {
                    std::cerr << "[control] MPRIS object failed: "
                              << (error == nullptr ? "unknown"
                                                  : error->message)
                              << "\n";
                    g_clear_error(&error);
                } else {
                    owner_subscription_ =
                        g_dbus_connection_signal_subscribe(
                            connection_, "org.freedesktop.DBus",
                            "org.freedesktop.DBus", "NameOwnerChanged",
                            "/org/freedesktop/DBus", kBluezService,
                            G_DBUS_SIGNAL_FLAGS_NONE,
                            bluez_owner_changed, this, nullptr);
                    register_with_bluez();

                    GSource* retry = g_timeout_source_new_seconds(3);
                    g_source_set_callback(
                        retry, retry_registration, this, nullptr);
                    g_source_attach(retry, context);
                    g_source_unref(retry);
                    g_main_loop_run(loop);
                }
            }
        }

        unregister_from_bluez();
        if (connection_ != nullptr && owner_subscription_ != 0) {
            g_dbus_connection_signal_unsubscribe(
                connection_, owner_subscription_);
        }
        if (connection_ != nullptr && object_id_ != 0) {
            g_dbus_connection_unregister_object(connection_, object_id_);
        }
        if (introspection_ != nullptr) {
            g_dbus_node_info_unref(introspection_);
            introspection_ = nullptr;
        }
        if (connection_ != nullptr) {
            g_object_unref(connection_);
            connection_ = nullptr;
        }
        g_main_context_pop_thread_default(context);
        {
            const std::scoped_lock lock(context_mutex_);
            context_ = nullptr;
            loop_ = nullptr;
        }
        g_main_loop_unref(loop);
        g_main_context_unref(context);
    }

    PauseControl& pause_control_;
    std::string controller_address_;
    std::string device_address_;
    std::string adapter_path_;
    std::thread worker_;
    std::atomic<bool> stop_requested_{false};
    std::mutex context_mutex_;
    GMainContext* context_ = nullptr;
    GMainLoop* loop_ = nullptr;
    GDBusConnection* connection_ = nullptr;
    GDBusNodeInfo* introspection_ = nullptr;
    guint object_id_ = 0;
    guint owner_subscription_ = 0;
    bool registered_ = false;
    bool adapter_missing_logged_ = false;
};

BluezMediaController::BluezMediaController(
    PauseControl& pause_control,
    std::string controller_address,
    std::string device_address)
    : impl_(std::make_unique<Impl>(
          pause_control, std::move(controller_address),
          std::move(device_address))) {}

BluezMediaController::~BluezMediaController() = default;

void BluezMediaController::start() {
    impl_->start();
}

void BluezMediaController::stop() noexcept {
    impl_->stop();
}

}  // namespace streamglass
