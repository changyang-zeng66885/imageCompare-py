import os
import sys
import time
import urllib
import requests
import re
from bs4 import BeautifulSoup
import time

header = {
    'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 UBrowser/6.1.2107.204 Safari/537.36'
}
url = "https://cn.bing.com/images/async?q={0}&first={1}&count={2}&scenario=ImageBasicHover&datsrc=N_I&layout=ColumnBased&mmasync=1&dgState=c*9_y*2226s2180s2072s2043s2292s2295s2079s2203s2094_i*71_w*198&IG=0D6AD6CBAF43430EA716510A4754C951&SFX={3}&iid=images.5599"


def getImage(url, count):
    '''从原图url中将原图保存到本地'''
    print(f"url:{url}")
    try:
        time.sleep(0.5)
        # urllib.request.urlretrieve(url, '/cloudsea' + str(count + 1) + '.jpg')
        save_image(url,"images/foguang/"+ str(count + 1) + '.jpg')
    except Exception as e:
        time.sleep(1)
        print("本张图片获取异常，跳过...")
    else:
        print("图片+1,成功保存 " + str(count + 1) + " 张图")

def save_image(image_url, save_path):
    try:
        # 发送请求获取图片
        response = requests.get(image_url,timeout=5)
        response.raise_for_status()  # 检查请求是否成功

        # 保存图片到指定路径
        with open(save_path, 'wb') as file:
            file.write(response.content)

        print(f"图片已保存至: {save_path}")
    except requests.exceptions.Timeout:
        print("下载超时，未下载该图片。")

    except Exception as e:
        print(f"下载失败: {e}")


def findImgUrlFromHtml(html, rule, url, key, first, loadNum, sfx, count):
    '''从缩略图列表页中找到原图的url，并返回这一页的图片数量'''
    soup = BeautifulSoup(html, "lxml")
    link_list = soup.find_all("a", class_="iusc")
    url = []

    for link in link_list:
        result = re.search(rule, str(link))
        # 将字符串"amp;"删除
        url = result.group(0)
        # 组装完整url
        url = url[8:len(url)]
        # 打开高清图片网址
        getImage(url, count)
        count += 1
    # 完成一页，继续加载下一页
    print(f"Find {len(url)} images")
    return count


def getStartHtml(url, key, first, loadNum, sfx):
    '''获取缩略图列表页'''
    page = urllib.request.Request(url.format(key, first, loadNum, sfx),
                                  headers=header)
    html = urllib.request.urlopen(page)
    return html


if __name__ == '__main__':
    name = "佛光 气象景观"  # 图片关键词
    countNum = 200 # 爬取数量
    key = urllib.parse.quote(name)
    first = 1
    loadNum = 1000
    sfx = 1
    count = 0
    rule = re.compile(r"\"murl\"\:\"http\S[^\"]+")
    while count < countNum:
        html = getStartHtml(url, key, first, loadNum, sfx)
        count = findImgUrlFromHtml(html, rule, url, key, first, loadNum, sfx,
                                   count)
        first = count + 1
        sfx += 1