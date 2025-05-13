import os

url = 'https://www.fon.hum.uva.nl/IFA-SpokenLanguageCorpora/IFADVcorpus/Speech/'
audio_file_url = f'{url}DVA4C.wav'

def download_audio_file(audio_file_url=audio_file_url):
    os.system(f'wget {audio_file_url}')

if __name__ == "__main__":
    try:download_audio_file()
    except Exception as e:
        print(f"An error occurred: {e}")
    else:
        print("Audio file downloaded successfully.")
