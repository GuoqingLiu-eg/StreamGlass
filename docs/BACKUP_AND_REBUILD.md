# 本地备份与重建

## 这份工程是什么

2026-10-11 从 Jetson AGX Orin 的 `/home/jetson/streamglass-deploy-staging` 拉取。
保留 C++/CUDA 源码、网页及素材、部署脚本、测试、完整第三方源码快照和原有许可证。
设备上的 `/opt/streamglass/current` 仍是早期 S0 发布目录；实际服务入口指向 staging，
因此此次以 staging 为源。设备源码没有 `.git`，这里初始化的是新的本地 Git 仓库。

本次没有更改设备上的源码或启动翻译服务，也没有生成新的 TensorRT 引擎。

## 本地原始备份与 Git 内容

原始压缩包位于仓库外：
`../coordination/streamglass_inspection_20261011/private-backup/streamglass-source-config-evidence.tar.gz`。
其内容包含完整原始源码、隐藏历史副本、设备配置和 106 个历史验收文件。
传输后已核对压缩包 SHA-256，原始文件不随仓库发布。

原始设备配置解压在同级备份目录的 `extracted/device-config/`，按设备绝对路径镜像保存。
目录包括 `/etc/streamglass`、用户 StreamGlass/PipeWire/WirePlumber 配置、
实际安装的 systemd 服务及 Bluetooth 服务覆盖配置。没有导出蓝牙配对密钥或 SSH 密钥。

Git 中保留：

- 当前源码、第三方源码、网页素材、构建和部署脚本。
- `deploy/device-config-example/`：设备配置示例，设备标识由占位符替代；使用前逐项填写。
- `config/platform.reference.json`：原平台软件版本参考，不含 machine-id、存储 UUID 或控制器标识。
- `config/models.manifest.json`、`config/third_party.lock.json`：模型清单和第三方版本记录。
- `evidence/jetson-2026-10-11/`：此次取回的历史验收记录，日期表示取回时间，不是测试时间。

`.gitignore` 排除原始设备配置、原始 `config/platform.lock.json`、本地历史副本及大规模生成物。
源码内现有的设备专用默认值仍然保留，以免改动运行行为；迁移到新眼镜时需要更新相关音频节点。

## 没有包含的大文件

不包含 Seamless 模型权重、导出的 ONNX、TensorRT engine、容器镜像、编译结果或录音。
`third_party/silero-vad-v6.2.1/` 中原本随源码快照保存的小型 VAD 模型保留，
因为当前 Silero 导出脚本需要它们；它们不属于 NVMe 上的大规模生成缓存。

第三方目录中的许可证保持原样；本次没有为自有代码添加或变更授权许可。

## 恢复现有设备

从本地原始备份恢复设备定制配置时，先检查用户名、项目路径、眼镜和蓝牙适配器标识。
重点保留原机的 AEC 180 ms 校准、Silero VAD 阈值以及 12800 ms 句长上限。
`config/runtime.yaml` 中的部分字段是早期参考配置，不能替代实际 `realtime.env` 和启动脚本。

配置示例不是自动安装包。安装 boot 服务的脚本会启用相关服务，应在音频设备及模型准备好后执行。
更换眼镜或蓝牙适配器后需要重新配对和校准 AEC；旧校准值不保证适合新硬件。

## 从源码重建的顺序

1. 准备 Jetson 平台依赖，参考 `config/platform.reference.json` 和项目原有部署说明。
2. 获取 `config/models.manifest.json` 中的原始模型及分词器，并用 `deploy/model_manifest.py verify` 校验。
3. 在目标 Jetson 上运行 `python3 deploy/audit_platform.py --output config/platform.lock.json`，
   将新的 fingerprint 通过 `STREAMGLASS_PLATFORM_FINGERPRINT` 传给构建与运行脚本。
   原机的 fingerprint 与设备绑定，不应当作跨机器通用标识。
4. 用 `deploy/s0/` 建立官方参考环境；按 `deploy/export/` 的 capture、export、verify 脚本依次导出各模块。
   `run_runtime_assets_export.sh` 还依赖 T2U capture，应在相应参考捕获完成后运行。
5. 用 `deploy/trt/build_*` 在目标设备生成所需 engine；当前默认链路混合使用 FP16、mixed 和 FP32。
6. 运行 `deploy/build_native.sh` 编译 C++ 程序，再检查目标设备上的模块一致性、时延及全双工音频。
7. 填入设备配置、设置实际路径后，按项目原有 service 安装脚本部署网页和实时服务。

这些步骤是现有脚本的恢复顺序，不是一键重建已经验收的承诺。
本次没有做全新环境下的完整重建，也不保证 TensorRT 重建结果与所有历史实验引擎逐位一致。

## 历史证据的范围

`evidence/jetson-2026-10-11/tmp/soak/final-180.json` 的校验和与原 README 记录相符。
它在实时和压力模式下各重复 180 句短样本。223.57 ms 为该短样本测试中的最大推理耗时，
不能代表所有输入；长句及 encoder cache 的负结果见原有研究记录。

## 本机检查

文件拉取后检查压缩包完整性、关键源码是否齐全、Git 忽略规则及源代码中的敏感内容。
主机侧测试的结果记录在仓库外 `../coordination/streamglass_inspection_20261011/host-tests.log`。
GPU 推理、TensorRT 加载和物理眼镜效果需要回到 Jetson 验证。

此次 Mac 主机侧运行 94 项测试，92 项通过，2 项现有文本契约断言失败：

- `test_each_globe_shortcut_has_the_expected_flag_and_code` 仍要求 emoji 国旗，当前网页已用 SVG。
- `test_runtime_streams_real_partial_updates_over_sse` 仍要求 `interval: float = 0.05`，当前源码的默认值不同。

已将这两项测试及相关 HTML/Python 文件与原始备份逐字节核对，未被此次整理修改。
本次仅做备份与仓库准备，没有调整运行代码或修改测试来消除这些失败。
