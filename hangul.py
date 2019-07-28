import json
import re
import pandas

filename = 'D:/namuwiki170327/namuwiki_20170327.json'
#filename = 'D:/namuwiki160229/namuwiki_20160229.json'

with open(filename) as data_file:
    data = json.load(data_file)
#data = pandas.read_json(filename)
print("number of articles: ", len(data))

black_list_title = ['공지사항/차단 내역/통합본']

# Check some statistics of whole dataset
count_dict = {}
for article in data:
    if article['title'] in black_list_title:
        continue  # remove blacklist article

    #     if(len(article['text']) > 10000 and len(article['text']) < 11000):
    #         print(article)
    #         break

    if count_dict.get(len(article['text'])) == None:
        count_dict[len(article['text'])] = 1
    else:
        count_dict[len(article['text'])] = count_dict[len(article['text'])] + 1

MAX_ARTICLE_SIZE = max(count_dict.keys())

bucket_size = 1000
num_bucket = MAX_ARTICLE_SIZE // bucket_size + 1

#print('num_bucket:', num_bucket)

bucket_counts = [0] * num_bucket
for key, value in count_dict.items():
    index = key // bucket_size
    bucket_counts[index] = bucket_counts[index] + value

chinese = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)
japanese = re.compile(u'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf]', re.UNICODE)
hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')

def strip(text):
    text = re.sub(r"\{\{\{#\!html[^\}]*\}\}\}", '', text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)  # remove html
    text = re.sub(r"#redirect .*", '', text, flags=re.IGNORECASE)  # remove redirect
    text = re.sub(r"\[\[분류:.*", '', text)  # remove 분류
    text = re.sub(r"\[\[파일:.*", '', text)  # remove 파일
    text = re.sub(r"\* 상위 문서 ?:.*", '', text)  # remove 상위문서
    text = re.sub(r"\[youtube\(\w+\)\]", '', text, flags=re.IGNORECASE)  # remove youtube
    text = re.sub(r"\[include\(([^\]|]*)(\|[^]]*)?\]", r'\1', text, flags=re.IGNORECASE)  # remove include
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]|]+)\]\]", r'\1', text)  # remove link
    text = re.sub(r"\[\*([^\]]*)\]", '', text)  # remove 각주
    text = re.sub(r"\{\{\{([^\ }|]*) ([^\}|]*)\}\}\}", r'\2', text)  # remove text color/size
    text = re.sub(r"'''([^']*)'''", r'\1', text)  # remove text bold
    text = re.sub(r"(~~|--)([^']*)(~~|--)", '', text)  # remove strike-through

    text = re.sub(r"\|\|(.*)\|\|", '', text)  # remove table

    text = chinese.sub('', text)  # remove chinese
    text = japanese.sub('', text)  # remove japanese
    #text = hangul.sub('', text) # remove exception hangul
    return text

MIN_TEXT_SIZE = 5000
count = 10
with open('input.txt', 'w', encoding = "utf-8") as f:
    for article in data:
        if len(article['text']) < MIN_TEXT_SIZE or len(article['text']) >= MAX_ARTICLE_SIZE:
            continue # skip too small, too large articles

        text = strip(article['text'])
        f.write("%s\n%s\n\n\n" % (article['title'], text))