from gtts import gTTS

falas = {
    "audio_go.mp3":   "Pode atravessar com cuidado.",
    "audio_stop.mp3": "Aguarde. Sinal vermelho.",
    "audio_off.mp3":  "Semáforo não identificado."
}

for fname, text in falas.items():
    tts = gTTS(text=text, lang='pt-br')
    tts.save(fname)
    print("gerado:", fname)
