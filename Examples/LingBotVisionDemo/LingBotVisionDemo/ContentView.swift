import SwiftUI
import UniformTypeIdentifiers

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
}
