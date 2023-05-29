import English_STT
import tokenisation

txt = English_STT.STT()
print(txt)
print("extracting symptoms and diseases from the text...")
symptoms = tokenisation.tokenisation(txt)
print("symptoms and diseases extracted from the text:", symptoms)