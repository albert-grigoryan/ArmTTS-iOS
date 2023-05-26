import SwiftUI
import ArmTTS

extension String {
  func trunc(length: Int, trailing: String = "") -> String {
    return (self.count > length) ? self.prefix(length) + trailing : self
  }
}

struct ContentView: View {
    @StateObject var tts = ArmTTS(X_RapidAPI_Key: "UPDATE_TOKEN_HERE")!
    
    @State var text = "Ողջույն, իմ անունը Գոռ է։"
    @State private var speed = 1.0
    
    let themeColor = Color(red: 0.26, green: 0.85, blue: 0.72)
    
    var body: some View {
        VStack {
            TextEditor(text: $text)
            HStack {
                Label("", systemImage: "tortoise")
                    .foregroundColor(themeColor)
                Slider(
                    value: $speed,
                    in: 0.5...1.5,
                    step: 0.01
                )
                Label("", systemImage: "hare")
                    .foregroundColor(themeColor)
            }
            Button {
                tts.speak(text: text, speed: Float(speed))
            } label: {
                Label("Speak", systemImage: "speaker.1")
                    .foregroundColor(themeColor)
            }
        }
        .padding()
    }
    
}

extension ArmTTS: ObservableObject {
    
}
