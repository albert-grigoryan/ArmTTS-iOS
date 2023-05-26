import Accelerate
import AVFoundation
import CoreImage
import Darwin
import Foundation
import UIKit
import onnxruntime_objc

struct ProcessResult {
    let sequence: [Int32];
    let rc: Int;
    let message: String;
}

func printError(_ msg: String) {
    print("[ArmTTS] Error: \(msg)")
}

public class ArmTTS : NSObject {
    private let API_URL = "https://armtts1.p.rapidapi.com/v2/preprocess"
    private let MODEL = "arm-gor"
    private let MAX_LENGTH = 140;
    private let SAMPLE_RATE = 44_100
    
    private let rapid_api_key: String
    private let sampleBufferRenderSynchronizer = AVSampleBufferRenderSynchronizer()
    private let sampleBufferAudioRenderer = AVSampleBufferAudioRenderer()
    
    private var session: ORTSession
    private var env: ORTEnv
    private var isInitialized: Bool = false
    
    func checkInit() -> Int {
        if self.isInitialized {
            return 0;
        }
        return 1;
    }
    
    func makeInputTensor(ids: [Int32]) throws -> ORTValue {
        let inputShape: [NSNumber] = [1, ids.count as NSNumber]
        let data = NSMutableData()
        for id in ids {
            var longId = Int64(id)
            let longData = Data(bytes: &longId, count: MemoryLayout.size(ofValue: longId))
            data.append(longData)
        }
        
        let tensor = try ORTValue(tensorData: data, elementType: ORTTensorElementDataType.int64, shape: inputShape)
        return tensor
    }
    
    func makeInputLengthsTensor(ids: [Int32]) throws -> ORTValue {
        let inputShape: [NSNumber] = [1]
        let data = NSMutableData()
        var longCount = Int64(ids.count)
        let longData = Data(bytes: &longCount, count: MemoryLayout.size(ofValue: longCount))
        data.append(longData)
        
        let tensor = try ORTValue(tensorData: data, elementType: ORTTensorElementDataType.int64, shape: inputShape)
        return tensor
    }
    
    func makeScales(speed: Float) throws -> ORTValue {
        let inputShape: [NSNumber] = [3]
        let data = NSMutableData()
        let scales: [Float32] = [0.0, speed, 0.0]
        for scale in scales {
            var tmp: Float32 = scale;
            let floatData = Data(bytes: &tmp, count: MemoryLayout.size(ofValue: tmp))
            data.append(floatData)
        }
        
        let tensor = try ORTValue(tensorData: data, elementType: ORTTensorElementDataType.float, shape: inputShape)
        return tensor
    }

    func synthesize(ids: [Int32], speed: Float) -> Data {
        do {
            let inputTensor = try makeInputTensor(ids: ids)
            let inputLengthsTensor = try makeInputLengthsTensor(ids: ids)
            let inputScalesTensor = try makeScales(speed: speed)
            
            // Run ORT Inference Session
            let inputs = ["input": inputTensor, "input_lengths": inputLengthsTensor, "scales": inputScalesTensor]
            let outputs = try session.run(withInputs: inputs,
                                          outputNames: ["output"],
                                          runOptions: nil)
            
            guard let rawOutputValue = outputs["output"] else {
                fatalError("failed to get model output")
            }
            let rawOutputData = try rawOutputValue.tensorData() as Data
            return rawOutputData;
        } catch {
            printError("Synthesize failed: \(error).")
        }
        return Data();
    }
    
    public init?(X_RapidAPI_Key: String) {
        self.rapid_api_key = X_RapidAPI_Key
        sampleBufferRenderSynchronizer.addRenderer(sampleBufferAudioRenderer)
        do {
            let myBundle = Bundle(for: Self.self)
            guard let resourceBundleURL = myBundle.url( forResource: "ArmTTS", withExtension: "bundle") else { fatalError("ArmTTS.bundle not found!") }
            guard let resourceBundle = Bundle(url: resourceBundleURL)
                else { fatalError("Cannot access ArmTTS.bundle!") }
            
            // Load Models
            if let modelPath: URL = resourceBundle.url(forResource: MODEL, withExtension: "onnx") {
                // Start the ORT inference environment and specify the options for session
                self.env = try ORTEnv(loggingLevel: ORTLoggingLevel.verbose)
                let options = try ORTSessionOptions()
                
                // Create the ORTSession
                self.session = try ORTSession(env: env, modelPath: modelPath.path, sessionOptions: options)
                self.isInitialized = true
            } else {
                printError("'\(MODEL).onnx' is missing.")
                return nil
            }
        } catch {
            printError("Initialization failed: \(error)\nPlease check the model (\(MODEL).onnx, is copied properly.")
            return nil
        }
        super.init()
    }
    
    private func tokenize(text: String) -> [String] {
        var tokens = [String]()
        let s_full_stops = try! NSRegularExpression(pattern: "[;։․.]")
        let s_commas = try! NSRegularExpression(pattern: "[՝`]")
        var modifiedText = s_full_stops.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.count), withTemplate: ":")
        modifiedText = s_commas.stringByReplacingMatches(in: modifiedText, options: [], range: NSRange(location: 0, length: modifiedText.count), withTemplate: ",")
        
        while modifiedText.count > 0 {
            if modifiedText.count <= MAX_LENGTH {
                tokens.append(modifiedText.trimmingCharacters(in: .whitespacesAndNewlines))
                break
            } else if let range = modifiedText.prefix(MAX_LENGTH).range(of: ":", options: .backwards) {
                let idx = modifiedText.distance(from: modifiedText.startIndex, to: range.lowerBound)
                tokens.append(modifiedText.prefix(idx + 1).trimmingCharacters(in: .whitespacesAndNewlines))
                modifiedText = String(modifiedText.suffix(from: range.upperBound))
            } else if let range = modifiedText.prefix(MAX_LENGTH).range(of: ",", options: .backwards) {
                let idx = modifiedText.distance(from: modifiedText.startIndex, to: range.lowerBound)
                tokens.append(modifiedText.prefix(idx + 1).trimmingCharacters(in: .whitespacesAndNewlines))
                modifiedText = String(modifiedText.suffix(from: range.upperBound))
            } else if let range = modifiedText.prefix(MAX_LENGTH).range(of: " ") {
                let idx = modifiedText.distance(from: modifiedText.startIndex, to: range.lowerBound)
                tokens.append(modifiedText.prefix(idx + 1).trimmingCharacters(in: .whitespacesAndNewlines) + ",")
                modifiedText = String(modifiedText.suffix(from: range.upperBound))
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
            data.append(synthesize(ids: processResult.sequence, speed: reversedSpeed))
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
            printError(String(describing: error))
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
