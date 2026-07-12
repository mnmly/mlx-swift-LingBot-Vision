import SwiftUI
import UniformTypeIdentifiers
#if canImport(AppKit)
import AppKit
#endif
#if canImport(Darwin)
import Darwin
#endif

@main
struct LingBotVisionDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        #if os(macOS)
        .defaultSize(width: 820, height: 640)
        .windowResizability(.contentMinSize)
        #endif
    }
}

struct ContentView: View {
    @State private var model = DemoModel()
    @State private var showModelPicker = false
    @State private var showImagePicker = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Critically damped, no overshoot — an appearing result isn't a momentum
    /// gesture, so it should settle without bounce. Nil under Reduce Motion.
    private var settle: Animation? {
        reduceMotion ? nil : .spring(response: 0.38, dampingFraction: 1.0)
    }

    var body: some View {
        VStack(spacing: 16) {
            header
            controlBar
            if model.isDownloading { downloadBar }
            panes
            statusBar
        }
        .padding(20)
        .frame(minWidth: 720, minHeight: 580)
        .animation(settle, value: model.isDownloading)
        .fileImporter(isPresented: $showModelPicker, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result { model.setModelDirectory(url) }
        }
        .fileImporter(isPresented: $showImagePicker, allowedContentTypes: [.image]) { result in
            if case .success(let url) = result { model.setImage(url) }
        }
    }

    // MARK: - Header (wayfinding: what is this, which model is loaded)

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            VStack(alignment: .leading, spacing: 1) {
                Text("LingBot-Vision")
                    .font(.title2.weight(.semibold))
                    .tracking(-0.2)  // tighten large text
                Text("Patch-feature PCA · MLX on Apple Silicon")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            if let dir = model.modelDirectory {
                Label(dir.lastPathComponent, systemImage: "cube.box")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(dir.path)
            }
        }
    }

    // MARK: - Control bar (grouped: model source · image · options · Run)

    private var controlBar: some View {
        HStack(spacing: 12) {
            HStack(spacing: 8) {
                Button { showModelPicker = true } label: {
                    Label("Folder", systemImage: "folder")
                }
                #if os(macOS)
                Button { chooseDownloadFolderAndDownload() } label: {
                    Label("Download", systemImage: "arrow.down.circle")
                }
                .disabled(model.isDownloading || model.isRunning)
                .help("Download the converted weights from Hugging Face")
                #endif
            }

            groupDivider

            Button { showImagePicker = true } label: {
                Label("Image", systemImage: "photo")
            }

            groupDivider

            Toggle("float16", isOn: $model.useFloat16)
                .toggleStyle(.switch)
                .help("fp16 compute (faster) vs fp32")
            Picker("Size", selection: $model.imageSize) {
                Text("256").tag(256)
                Text("384").tag(384)
                Text("512").tag(512)
            }
            .labelsHidden()
            .frame(width: 92)
            .help("Input resolution")

            Spacer(minLength: 8)

            Button { model.run() } label: {
                HStack(spacing: 6) {
                    if model.isRunning {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: "play.fill")
                    }
                    Text("Run").fontWeight(.semibold)
                }
                .frame(minWidth: 52)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .keyboardShortcut(.return, modifiers: [])
            .disabled(!model.canRun)
        }
        .buttonStyle(.bordered)
        .controlSize(.large)
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.white.opacity(0.07))  // faint top-light on the material
        )
    }

    private var groupDivider: some View {
        Divider().frame(height: 18)
    }

    // MARK: - Download progress

    private var downloadBar: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Label("Downloading weights", systemImage: "arrow.down.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(downloadDetail)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            // Indeterminate until the large file reports its size, so the bar
            // animates ("working") from the first moment rather than sitting at 0%.
            if model.downloadIndeterminate {
                ProgressView().progressViewStyle(.linear)
            } else {
                ProgressView(value: model.downloadProgress).progressViewStyle(.linear)
            }
        }
        .padding(12)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .transition(.opacity.combined(with: .move(edge: .top)))
    }

    private var downloadDetail: String {
        let mb = Double(model.downloadedBytes) / 1_048_576
        if model.downloadIndeterminate {
            return mb > 0 ? String(format: "%.0f MB", mb) : "starting…"
        }
        return "\(Int(model.downloadProgress * 100))%"
    }

    // MARK: - Image panes

    private var panes: some View {
        HStack(spacing: 16) {
            pane(model.inputImage,
                 label: "Input",
                 systemImage: "photo",
                 placeholder: "Choose an image",
                 busy: false,
                 token: model.inputImage == nil ? 0 : 1)
            pane(model.pcaImage,
                 label: "Patch PCA",
                 systemImage: "sparkles",
                 placeholder: "Run to visualize features",
                 busy: model.isRunning,
                 token: model.resultToken)
        }
    }

    @ViewBuilder
    private func pane(_ image: CGImage?, label: String, systemImage: String,
                      placeholder: String, busy: Bool, token: Int) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: systemImage).foregroundStyle(.secondary)
                Text(label).font(.headline)
            }
            ZStack {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(.quaternary)

                if let image {
                    // A fresh `.id` per result re-inserts the view so the
                    // materialize transition replays each Run, not just the first.
                    Image(image, scale: 1, label: Text(label))
                        .resizable()
                        .interpolation(.none)
                        .aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                        .id(token)
                        .transition(
                            reduceMotion
                                ? .opacity
                                : .scale(scale: 0.96).combined(with: .opacity))
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: systemImage)
                            .font(.system(size: 30, weight: .light))
                            .foregroundStyle(.tertiary)
                        Text(placeholder)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                }

                if busy {
                    // Dim-to-focus while computing.
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(.ultraThinMaterial)
                    ProgressView().controlSize(.large)
                }
            }
            .frame(maxWidth: .infinity)
            .aspectRatio(1, contentMode: .fit)
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(.quaternary, lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.10), radius: 10, y: 3)
            .animation(settle, value: token)
            .animation(settle, value: busy)
        }
    }

    // MARK: - Status (feedback kind: info / success / error)

    private var statusBar: some View {
        HStack(spacing: 8) {
            Image(systemName: statusIcon)
                .foregroundStyle(statusTint)
                .font(.callout)
            Text(model.status)
                .font(.callout)
                .foregroundStyle(.secondary)
                .lineLimit(2)
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .animation(settle, value: model.statusKind)
    }

    private var statusIcon: String {
        switch model.statusKind {
        case .info: return "info.circle"
        case .success: return "checkmark.circle.fill"
        case .error: return "exclamationmark.triangle.fill"
        }
    }

    private var statusTint: Color {
        switch model.statusKind {
        case .info: return .secondary
        case .success: return .green
        case .error: return .orange
        }
    }

    // MARK: - Download flow (sandbox-aware folder picker)

    #if os(macOS)
    /// Present a folder picker (defaulting to the real `~/.cache/huggingface`)
    /// and download the weights into the chosen folder. The picker is required:
    /// the App Sandbox only grants write access to a user-selected location.
    private func chooseDownloadFolderAndDownload() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Download Here"
        panel.message = "Choose a folder to save the model (e.g. ~/.cache/huggingface)."
        let cache = realHomeDirectory().appendingPathComponent(".cache/huggingface", isDirectory: true)
        if FileManager.default.fileExists(atPath: cache.path) {
            panel.directoryURL = cache
        }
        if panel.runModal() == .OK, let url = panel.url {
            model.downloadFromHub(into: url)
        }
    }

    /// The user's real home directory. Under the App Sandbox,
    /// `FileManager.homeDirectoryForCurrentUser` returns the container path,
    /// but `getpwuid` still reports the true home — so the picker can default to
    /// the standard Hugging Face cache outside the container.
    private func realHomeDirectory() -> URL {
        if let pw = getpwuid(getuid()), let home = pw.pointee.pw_dir {
            return URL(fileURLWithPath: String(cString: home))
        }
        return FileManager.default.homeDirectoryForCurrentUser
    }
    #endif
}
