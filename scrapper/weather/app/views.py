from django.shortcuts import render
import requests
from bs4 import BeautifulSoup


def get_html_content(request):
    city = request.GET.get('city')
    city = city.replace(" ", "+")

    USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"
    LANGUAGE = "en-US,en;q=0.5"
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT
    session.headers['Accept-Language'] = LANGUAGE
    session.headers['Content-Language'] = LANGUAGE

    content = session.get(f'https://www.google.com/search?q=weather+{city}').text
    return content


def home(request):
    weather_data = None
    if 'city' in request.GET:
        content = get_html_content(request)

        soup = BeautifulSoup(content, 'html.parser')
        weather_data = dict()

        weather_data['region'] = soup.find("span", attrs={"class": "BNeawe tAd8D AP7Wnd"}).text
        weather_data['temp'] = soup.find("div", attrs={"class": "BNeawe iBp4i AP7Wnd"}).text
        weather_data['time_now'], weather_data['condition'] = soup.find("div", attrs={"class": "BNeawe tAd8D AP7Wnd"}).text.split('\n')

    return render(request, 'app/home.html', {'result': weather_data})
