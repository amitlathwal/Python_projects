import json
import requests

num = 2
api_url_base = 'https://youtube.googleapis.com/youtube/v3/search?publishedBefore=2021-01-01T00%3A00%3A00Z'
data = {
        'maxResults':num,
        'part':'snippet',
        'order':'rating',
        'type':'video',
        'key':'AIzaSyCF4bzZdDolvL_mTVAehSguwyPB35N5dVs',
    }

response = requests.get(api_url_base,params=data)
out = response.json()
status = response.status_code

try:
    if status != 200:
        raise ValueError
    else:
        for i in range(0,num):
            videoID = out['items'][i]['id']['videoId']
            title = out['items'][i]['snippet']['title']
            url = out['items'][i]['snippet']['thumbnails']['high']['url']
            print('videoID: ' + videoID)
            print('title: ' + title)
            print('url: ' + str(url))
            print('')

except ValueError:
    print('Bad parameters for API.')
except:
    print('Not able to interact with API.')