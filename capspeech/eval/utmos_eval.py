import utmosv2
model = utmosv2.create_model(pretrained=True)
mos = model.predict(input_path="your-audio-path")
print(mos)