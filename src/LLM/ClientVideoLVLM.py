import requests
LVLM_API_URL = 'http://127.0.0.1:5000/'

class ClienVideoLVLM:
	def __init__(self, proxies: dict = None) -> None:
		if proxies is None:
			print('Running Video LVLM takes a lot of performance')
		self.proxies = proxies

	def generate_simple_response(self, prompt: str, video_path: str) -> str:
		self._upload_video(video_path)
		
		payload = {
			'prompt': prompt
		}
		url = LVLM_API_URL
		if self.proxies is not None:
			response = requests.post(url, json=payload, proxies=self.proxies)
		else:
			print('Running Video LVLM takes a lot of performance, not implemented yet')
			response = requests.post(url, json=payload)
		return response.text

	def _upload_video(self, video_path: str) -> None:
		url = LVLM_API_URL + "/upload"

		with open(video_path, 'rb') as f:
			files = {'file': f}
			if self.proxies is not None:
				response = requests.post(url, files=files, proxies=self.proxies)
			else:
				print('Running Video LVLM takes a lot of performance, not implemented yet')
				response = requests.post(url, files=files)
		print(response.text)

if __name__ == '__main__':
	proxies = {
        "http"  : "socks5h://localhost:1080",
        "https" : "socks5h://localhost:1080",
    }
	clientVideo = ClienVideoLVLM(proxies)
	print(clientVideo.generate_simple_response('In this video, an object is in motion. Describe the motion of the object', 'records/Hopper-v5/rl-video-episode-0.mp4'))