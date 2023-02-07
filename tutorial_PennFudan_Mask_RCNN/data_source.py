import requests
url ='https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
r = requests.get(url,stream=True)
with open('PennFudanPed.zip','wb+') as f:
    for chunk in r.iter_content(1024**2):
        f.write(chunk)