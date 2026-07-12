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
        .defaultSize(width: 760, height: 520)
        #endif
    }
}

struct ContentView: View {
    @State private var model = DemoModel()
    @State private var showModelPicker = false
    @State private var showImagePicker = false

    var body: some View {
        VStack(spacing: 14) {
            controls
            if model.isDownloading {
                ProgressView(value: model.downloadProgress) {
                    Text("Downloading weights… \(Int(model.downloadProgress * 100))%")
                        .font(.caption)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            HStack(spacing: 14) {
                pane(model.inputImage, label: "Input")
                pane(model.pcaImage, label: "Patch PCA")
            }
            Text(model.status)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .frame(minWidth: 720, minHeight: 480)
        .fileImporter(isPresented: $showModelPicker, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result { model.setModelDirectory(url) }
        }
        .fileImporter(isPresented: $showImagePicker, allowedContentTypes: [.image]) { result in
            if case .success(let url) = result { model.setImage(url) }
        }
    }

    private var controls: some View {
        HStack(spacing: 10) {
            Button("Model Folder…") { showModelPicker = true }
            #if os(macOS)
            Button("Download from HF…") { chooseDownloadFolderAndDownload() }
                .disabled(model.isDownloading || model.isRunning)
            #endif
            Button("Image…") { showImagePicker = true }
            Toggle("float16", isOn: $model.useFloat16)
                .toggleStyle(.switch)
            Picker("Size", selection: $model.imageSize) {
                Text("256").tag(256)
                Text("384").tag(384)
                Text("512").tag(512)
            }
            .frame(width: 150)
            Spacer()
            Button(action: { model.run() }) {
                if model.isRunning {
                    ProgressView().controlSize(.small)
                } else {
                    Text("Run").bold()
                }
            }
            .keyboardShortcut(.return, modifiers: [])
            .disabled(!model.canRun)
        }
    }

    @ViewBuilder
    private func pane(_ image: CGImage?, label: String) -> some View {
        VStack(spacing: 6) {
            Text(label).font(.headline)
            ZStack {
                RoundedRectangle(cornerRadius: 8).fill(.quaternary)
                if let image {
                    Image(image, scale: 1, label: Text(label))
                        .resizable()
                        .interpolation(.none)
                        .aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    Text("—").foregroundStyle(.secondary)
                }
            }
            .frame(width: 320, height: 320)
        }
    }

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
