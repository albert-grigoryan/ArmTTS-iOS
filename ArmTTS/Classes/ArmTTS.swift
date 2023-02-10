import Foundation
import AVFoundation
import TensorFlowLite

struct ProcessResult {
    let sequence: [Int32];
    let rc: Int;
    let message: String;
}

public class ArmTTS {
    private let API_URL = "https://armtts1.p.rapidapi.com/preprocess"
    private let MODEL_1 = "model1"
    private let MODEL_2 = "model2"
    private let MAX_LENGTH = 160;
    private let SAMPLE_RATE = 22_000
    
    private let rapid_api_key: String
    private var interpreter1: Interpreter?
    private var interpreter2: Interpreter?
    private let sampleBufferRenderSynchronizer = AVSampleBufferRenderSynchronizer()
    private let sampleBufferAudioRenderer = AVSampleBufferAudioRenderer()
    
    func printError(_ msg: String) {
        print("[ArmTTS] Error: \(msg)")
    }
    
    func checkInit() -> Int {
        if interpreter1 == nil || interpreter2 == nil {
            return 1;
        }
        return 0;
    }
    
    func initModels() {
        do {
            let myBundle = Bundle(for: Self.self)
            guard let resourceBundleURL = myBundle.url( forResource: "ArmTTS", withExtension: "bundle") else { fatalError("ArmTTS.bundle not found!") }
            guard let resourceBundle = Bundle(url: resourceBundleURL)
                else { fatalError("Cannot access ArmTTS.bundle!") }
            
            // Load Models
            if let url: URL = resourceBundle.url(forResource: MODEL_1, withExtension: "tflite") {
                var options = Interpreter.Options()
                options.threadCount = 5
                interpreter1 = try Interpreter(modelPath: url.path, options: options)
            } else {
                printError("'\(MODEL_1).tflite' is missing.")
            }
            if let url: URL = resourceBundle.url(forResource: MODEL_2, withExtension: "tflite") {
                var options = Interpreter.Options()
                options.threadCount = 5
                interpreter2 = try Interpreter(modelPath: url.path, options: options)
            } else {
                printError("'\(MODEL_2).tflite' is missing.")
            }
        } catch {
            printError("Initialization failed: \(error)\nPlease check the models (\(MODEL_1).tflite, \(MODEL_2).tflite) are copied properly.")
        }
    }

    public init(X_RapidAPI_Key: String) {
        self.rapid_api_key = X_RapidAPI_Key
        sampleBufferRenderSynchronizer.addRenderer(sampleBufferAudioRenderer)
        initModels()
    }
    
    func split(str: String, chars: String) -> [String] {
        var result: [String] = []
        let tmp = str.components(separatedBy: CharacterSet(charactersIn: chars))
        for token in tmp {
            let t = token.trimmingCharacters(in: .whitespaces)
            if (!t.isEmpty) {
                result.append(t)
            }
        }
        return result
    }
    
    private func tokenize(text: String) -> [String] {
        let tmp_tokens = split(str: text, chars: "։:.")
        var tokens: [String] = []
        for token in tmp_tokens {
            if (token.count > MAX_LENGTH) {
                var tmp_string = ""
                for word in split(str: token, chars: " ") {
                    if (tmp_string.count + word.count < MAX_LENGTH/2) {
                        tmp_string += " " + word;
                    } else {
                        if (!tmp_string.isEmpty) {
                            tokens.append(String(tmp_string.trimmingCharacters(in: .whitespaces)))
                        }
                        tmp_string = word
                    }
                }
                if (!tmp_string.isEmpty) {
                    tokens.append(String(tmp_string.trimmingCharacters(in: .whitespaces)))
                }
            } else {
                tokens.append(String(token.trimmingCharacters(in: .whitespaces)))
            }
        }
        return tokens
    }
    
    private func play(data: Data) {
        do {
            let blockBuffer = try CMBlockBuffer(length: data.count)
            try data.withUnsafeBytes { try blockBuffer.replaceDataBytes(with: $0) }
            let audioStreamBasicDescription = AudioStreamBasicDescription(mSampleRate: Float64(SAMPLE_RATE), mFormatID: kAudioFormatLinearPCM, mFormatFlags: kAudioFormatFlagIsFloat, mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4, mChannelsPerFrame: 1, mBitsPerChannel: 32, mReserved: 0)
            let formatDescription = try CMFormatDescription(audioStreamBasicDescription: audioStreamBasicDescription)
            let delay: TimeInterval = 1
            let sampleBuffer = try CMSampleBuffer(dataBuffer: blockBuffer, formatDescription: formatDescription, numSamples: data.count / 4, presentationTimeStamp: sampleBufferRenderSynchronizer.currentTime() + CMTime(seconds: delay, preferredTimescale: CMTimeScale(SAMPLE_RATE)), packetDescriptions: [])
            sampleBufferAudioRenderer.enqueue(sampleBuffer)
            sampleBufferRenderSynchronizer.rate = 1
        }
        catch {
            printError(error.localizedDescription)
        }
    }
    
    private func synthesize(input_ids: [Int32], speed: Float = 1.0) -> Data {
        do {
            try interpreter1!.resizeInput(at: 0, to: [1, input_ids.count])
            try interpreter1!.allocateTensors()
            let data = input_ids.withUnsafeBufferPointer(Data.init)
            try interpreter1!.copy(data, toInputAt: 0)
            var speakerId: Int32 = 0
            try interpreter1!.copy(Data(bytes: &speakerId, count: 4), toInputAt: 1)
            var speedRatio = speed
            try interpreter1!.copy(Data(bytes: &speedRatio, count: 4), toInputAt: 2)
            var f0Ratio: Float = 1
            try interpreter1!.copy(Data(bytes: &f0Ratio, count: 4), toInputAt: 3)
            var energyRatio: Float = 1
            try interpreter1!.copy(Data(bytes: &energyRatio, count: 4), toInputAt: 4)
            try interpreter1!.invoke()
            let melSpectrogram = try interpreter1!.output(at: 1)
            try interpreter2!.resizeInput(at: 0, to: melSpectrogram.shape)
            try interpreter2!.allocateTensors()
            try interpreter2!.copy(melSpectrogram.data, toInputAt: 0)
            try interpreter2!.invoke()
            let data2 = try interpreter2!.output(at: 0).data
            return data2
        } catch {
            printError(error.localizedDescription)
        }
        return Data();
    }

    public func speak(text: String, speed: Float = 1.0) {
        let reversedSpeed = 2 - speed
        if checkInit() != 0 {
            printError("The initialization has been failed. Please check the logs for more details.")
            return;
        }
        var data: Data = Data()
        let tokens: [String] = tokenize(text: text)
        for token in tokens {
            let processResult = process(token)
            if (processResult.rc != 0) {
                printError(processResult.message)
                return
            }
            data.append(synthesize(input_ids: processResult.sequence, speed: reversedSpeed))
        }
        play(data: data)
    }
    
    func process(_ text: String) -> ProcessResult {
        var sequence: [Int32] = []
        var message: String = "";
        var rc: Int = 0;
        let semaphore = DispatchSemaphore (value: 0)
        let parameters = [
          ["key": "text", "value": text, "type": "text"]] as [[String : Any]]
        let boundary = "Boundary-\(UUID().uuidString)"
        var body = ""
        for param in parameters {
          if param["disabled"] == nil {
            let paramName = param["key"]!
            body += "--\(boundary)\r\n"
            body += "Content-Disposition:form-data; name=\"\(paramName)\""
            if param["contentType"] != nil {
              body += "\r\nContent-Type: \(param["contentType"] as! String)"
            }
            let paramType = param["type"] as! String
            if paramType == "text" {
              let paramValue = param["value"] as! String
              body += "\r\n\r\n\(paramValue)\r\n"
            } else {
              let paramSrc = param["src"] as! String
              let fileData = try! NSData(contentsOfFile:paramSrc, options:[]) as Data
              let fileContent = String(data: fileData, encoding: .utf8)!
              body += "; filename=\"\(paramSrc)\"\r\n"
                + "Content-Type: \"content-type header\"\r\n\r\n\(fileContent)\r\n"
            }
          }
        }
        body += "--\(boundary)--\r\n";
        let postData = body.data(using: .utf8)
        var request = URLRequest(url: URL(string: API_URL)!,timeoutInterval: Double.infinity)
        request.addValue(rapid_api_key, forHTTPHeaderField: "X-RapidAPI-Key")
        request.addValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpMethod = "POST"
        request.httpBody = postData
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
          guard let data = data else {
            self.printError(String(describing: error))
            semaphore.signal()
            return
          }
          let json = try! JSONSerialization.jsonObject(with:data, options :[]) as! [String:Any]
          if let ids = json["ids"] as? [Int32] {
            sequence = ids;
          } else {
              rc = -1;
              message = json["message"] as? String ?? ""
          }
          semaphore.signal()
        }
        task.resume()
        semaphore.wait()
        let result = ProcessResult(sequence: sequence, rc: rc, message: message)
        return result;
    }
}
