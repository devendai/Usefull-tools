import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

txtfile = 'visual_aids.txt'
alice_novel = open(txtfile, 'r').read()
stopwords = set(STOPWORDS)

# instantiate a word cloud object
alice_wc = WordCloud(
    background_color='white',
    max_words=10000,
    stopwords=stopwords
)

# generate the word cloud
alice_wc.generate(alice_novel)

fig = plt.figure()
fig.set_figwidth(16) # set width
fig.set_figheight(6) # set height

# display the cloud
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')

# saving figure as png
outFile = 'wordCld.png'
plt.savefig(outFile)
plt.close()