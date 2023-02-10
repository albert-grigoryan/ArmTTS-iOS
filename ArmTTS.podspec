Pod::Spec.new do |s|
  s.name             = 'ArmTTS'
  s.version          = '1.0.0'
  s.summary          = 'Text-to-Speech (TTS) engine for the Armenian language.'

  s.description      = <<-DESC
  Text-to-Speech (TTS) engine for the Armenian language designed by Albert Grigoryan.
                       DESC

  s.homepage         = 'https://armtts.online'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'Albert Grigoryan' => 'albert_grigoryan@yahoo.com' }
  s.source           = { :git => 'https://github.com/albert-grigoryan/ArmTTS-iOS.git', :tag => s.version.to_s }

  s.swift_versions = ['5.0']
  s.ios.deployment_target = '13.0'

  s.source_files = 'ArmTTS/Classes/**/*'
  s.resource_bundles = {
     'ArmTTS' => ['ArmTTS/Assets/*.tflite']
  }
  s.static_framework = true
  s.pod_target_xcconfig = { 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64' }
  s.user_target_xcconfig = { 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64' }
  s.dependency 'TensorFlowLiteSwift', '~> 2.4.0'
  s.dependency 'TensorFlowLiteSelectTfOps', '~> 2.4.0'
end
