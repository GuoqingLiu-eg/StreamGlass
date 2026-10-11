-- StreamGlass runs under a lingering user manager before a graphical login.
-- Disable logind seat arbitration so that this dedicated user owns Bluetooth
-- audio at boot, and ask BlueZ to restore all headset profiles automatically.
bluez_monitor.properties["with-logind"] = false
bluez_monitor.properties["bluez5.enable-msbc"] = true

table.insert(bluez_monitor.rules, 1, {
  matches = {
    {
      { "device.name", "matches", "bluez_card.*" },
    },
  },
  apply_properties = {
    ["bluez5.auto-connect"] =
        "[ hfp_hf hsp_hs a2dp_sink hfp_ag hsp_ag a2dp_source ]",
  },
})

-- Keep the full-duplex headset nodes open while the translation runtime is
-- paused. The runtime still owns active capture/playback streams; disabling
-- session suspension prevents WirePlumber from treating a quiet interval as
-- a reason to close the underlying Bluetooth transport.
table.insert(bluez_monitor.rules, 1, {
  matches = {
    {
      { "node.name", "matches", "bluez_input.*" },
    },
    {
      { "node.name", "matches", "bluez_output.*" },
    },
  },
  apply_properties = {
    ["node.pause-on-idle"] = false,
    ["session.suspend-timeout-seconds"] = 0,
  },
})
