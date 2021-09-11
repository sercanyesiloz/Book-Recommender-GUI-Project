#%% Libraries and Utilities
import re
import nltk
import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
# from tkinter import ttk
from PIL import Image

from difflib import SequenceMatcher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
nltk.download('stopwords')
#%% Load Dataset
books_path = 'C:/Users/q/Desktop/Book Recommender GUI Project/books.csv'
books = pd.read_csv(books_path)
books.isnull().sum()
#%% Preprocessing

rating_df = books.copy()
rating_df.drop(index=rating_df[rating_df['Category'] == '9'].index, inplace=True)
rating_df.drop(index=rating_df[rating_df['rating'] == 0].index, inplace=True)

df = books.copy()
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df.drop(columns = ['Unnamed: 0','location','isbn',
                   'img_s','img_m','city','age',
                   'state'],axis=1,inplace = True) #remove useless cols

df.drop(index=df[df['Category'] == '9'].index, inplace=True) #remove 9 in category

df.drop(index=df[df['rating'] == 0].index, inplace=True) #remove 0 in rating

df['Category'] = df['Category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())

rating_counts = pd.DataFrame(df['book_title'].value_counts())
rare_books = rating_counts[rating_counts['book_title'] <= 100].index
df = df[~df['book_title'].isin(rare_books)]
df = df.drop_duplicates(subset = ['book_title'])
summary_filtered = []
for i in df['Summary']:
    
    i = re.sub("[^a-zA-Z]"," ",i).lower()
    i = nltk.word_tokenize(i)
    i = [word for word in i if not word in set(stopwords.words("english"))]
    i = " ".join(i)
    summary_filtered.append(i)

df['summary_filtered'] = summary_filtered
df.reset_index(drop = True, inplace =True)
print(df.shape)

#%% Functions

# Find Book
def find_book(book_title):
    
    seq = pd.DataFrame({'book': [i for i in df['book_title'].unique()],
                       'ratio': [SequenceMatcher(None, book_title, i.lower()). \
                                 ratio() for i in df['book_title'].unique()]}). \
    sort_values('ratio', ascending = False)
    
    seq = seq['book'].values[0]
    return seq

# Plotting selected book
def plot_book(book_title):

    fig, axs1 = plt.subplots(1, 1,figsize=(2,2.5))
    url = df.loc[df['book_title'] == book_title,'img_l'][:1].values[0]
    im = Image.open(requests.get(url, stream=True).raw)
    axs1.imshow(im)
    axs1.axis("off")
    axs1.set_title('Rating: {}'.format(round(rating_df[rating_df['book_title'] == book_title]['rating'].mean(),1)),
                 y=-0.18,
                     color="black",
                     fontsize=12)
    return fig

# Plotting all recommended books
def plot_5_books(recommended_books):
    
    fig, axs = plt.subplots(1, 5,figsize=(11.5,3))
    fig.suptitle('You may also like these books', size = 12)
    for i in range(len(recommended_books)):
        url = df.loc[df['book_title'] == recommended_books[i],'img_l'][:1].values[0]
        im = Image.open(requests.get(url, stream=True).raw)
        axs[i].imshow(im)
        axs[i].axis("off")
        axs[i].set_title('Rating: {}'.format(round(rating_df[rating_df['book_title'] == recommended_books[i]]['rating'].mean(),1)),
                     y=-0.18,
                         color="black",
                         fontsize=10)
    return fig

# Delete TextBoxes
def delete_textboxes():
    textBox.delete('1.0', 'end')
    textBox2.delete('1.0', 'end')
    textBox3.delete('1.0', 'end')
    textBox4.delete('1.0', 'end')
    textBox5.delete('1.0', 'end')
    textBox6.delete('1.0', 'end')
    textBox7.delete('1.0', 'end')
    
# Content-Based Recommender (3: (Title, Author, Publisher, Category), 2: Summary)
def content_based_recommender(book_title):
    
    #Content-Based CF (Title, Author, Publisher, Category)
    cv = CountVectorizer()
    book_data = df.copy()
    book_data.reset_index(inplace= True)
    book_data['index'] = [i for i in range(book_data.shape[0])]
    target_cols = ['book_title','book_author','publisher','Category']
    book_data['combined_features'] = [' '.join(book_data[target_cols].iloc[i,].values).lower() for i in range(book_data[target_cols].shape[0])]
    count_matrix = cv.fit_transform(book_data['combined_features'])
    cosine_sim = cosine_similarity(count_matrix)
    index = book_data[book_data['book_title'] == book_title]['index'].values[0]
    sim_books = list(enumerate(cosine_sim[index]))
    sorted_sim_books = sorted(sim_books,key=lambda x:x[1],reverse=True)[1:4]

    recom_books = []
    for i in sorted_sim_books:
        recom_books.append(book_data[book_data['index'] == i[0]]['book_title'].values[0])

    book_data = book_data[~book_data['book_title'].isin(recom_books)]
    book_data['index'] = [i for i in range(book_data.shape[0])]
    index = book_data[book_data['book_title'] == book_title]['index'].values[0]


#   Content-Based CF (Summary)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(book_data['summary_filtered'])
    cosine_sim = cosine_similarity(count_matrix) 
    sim_books = list(enumerate(cosine_sim[index]))
    sorted_sim_books = sorted(sim_books,key=lambda x:x[1],reverse=True)[1:3]
#     print('SORTED SIM: {}'.format(len(sorted_sim_books)))
#     print(sorted_sim_books)
    
    for i in sorted_sim_books:
        recom_books.append(book_data[book_data['index'] == i[0]]['book_title'].values[0])
    
    return recom_books

#%% Builting Frame

#Window
window = tk.Tk()
window.geometry('900x550')
window.title('Book Recommender')

# pw = ttk.Panedwindow(window, orient = tk.VERTICAL)
# pw.pack(fill = tk.BOTH, expand = True)

# Frames
frame2 = tk.Frame(window, width = 540, height = 640)
frame3 = tk.Frame(window, width = 200, height = 200)

# frame2 = ttk.Frame(pw, width = 900, height = 290, relief = tk.RIDGE)
# frame3 = ttk.Frame(pw, width = 900, height = 210, relief = tk.RIDGE)

frame4 = tk.LabelFrame(window, text = None, width = 270, height = 190)
frame4.place(x = 600, y = 85)

frame5 = tk.LabelFrame(window, text = None, width = 360, height = 190)
frame5.place(x = 220, y = 85)

frame6 = tk.LabelFrame(window, text = None, width = 160, height = 190)
frame6.place(x = 40, y = 85)

frame7 = tk.LabelFrame(window, text = None, width = 830, height = 220)
frame7.place(x = 40, y = 285)

# pw2 = ttk.Panedwindow(frame3, orient = tk.VERTICAL)
# pw2.pack(fill = tk.BOTH, expand = True)

# pw.add(frame1)
# pw.add(frame2)
# pw.add(frame3)

#Labels
label2 = tk.Label(window, text = 'Name:', font = 'Verdana 10',fg = 'black', wraplength = 200)
label2.place(x = 230, y = 95)

label3 = tk.Label(window, text = 'Author:', font = 'Verdana 10', fg = 'black', wraplength = 50)
label3.place(x = 230, y = 125)

label4 = tk.Label(window, text = 'Category:', font = 'Verdana 10',fg = 'black', wraplength = 80)
label4.place(x = 230, y = 155)
    
label5 = tk.Label(window, text = 'Year of Publication:', font = 'Verdana 10',fg = 'black', wraplength = 200)
label5.place(x = 230, y = 185)
    
label6 = tk.Label(window, text = 'Language:', font = 'Verdana 10',fg = 'black', wraplength = 80)
label6.place(x = 230, y = 215)
    
label7 = tk.Label(window, text = 'Country:', font = 'Verdana 10',fg = 'black', wraplength = 60)
label7.place(x = 230, y = 245)

label8 = tk.Label(window, text = 'Summary', font = 'Verdana 10',fg = 'black', wraplength = 60)
label8.place(x = 700, y = 90)

#Textboxes & Scrollbar
textBox = tk.Text(frame4, width = 25, height = 8, wrap = "word")
textBox.grid(row = 0, column = 0, padx =22, pady = 27)
scroll = tk.Scrollbar(frame4, orient = tk.VERTICAL, command = textBox.yview)
scroll.grid(row = 0, column = 1, sticky = tk.N + tk.S, pady=5)
textBox.config(yscrollcommand = scroll.set)
textBox.config(state='normal')

textBox2 = tk.Text(window, width = 24, height = 1, wrap = "char")
textBox2.place(x = 375, y = 95)
textBox2.config(state='normal')

textBox3 = tk.Text(window, width = 24, height = 1, wrap = "word")
textBox3.place(x = 375, y = 125)
textBox3.config(state='normal')

textBox4 = tk.Text(window, width = 24, height = 1, wrap = "word")
textBox4.place(x = 375, y = 155)
textBox4.config(state='normal')

textBox5 = tk.Text(window, width = 24, height = 1, wrap = "word")
textBox5.place(x = 375, y = 185)
textBox5.config(state='normal')

textBox6 = tk.Text(window, width = 24, height = 1, wrap = "word")
textBox6.place(x = 375, y = 215)
textBox6.config(state='normal')

textBox7 = tk.Text(window, width = 24, height = 1, wrap = "word")
textBox7.place(x = 375, y = 245)
textBox7.config(state='normal')

# textBox3 = tk.Text(frame5, width = 18, height = 1, wrap = "word")
# textBox3.grid(row = 1, column = 0, padx =100, pady = 5.5)
# textBox3.config(state='normal')

# textBox4 = tk.Text(frame5, width = 18, height = 1, wrap = "word")
# textBox4.grid(row = 2, column = 0, padx =100, pady = 5.5)
# textBox4.config(state='normal')

# textBox4 = tk.Text(frame5, width = 18, height = 1, wrap = "word")
# textBox4.grid(row = 3, column = 0, padx =100, pady = 5.5)
# textBox4.config(state='normal')

# textBox5 = tk.Text(frame5, width = 18, height = 1, wrap = "word")
# textBox5.grid(row = 4, column = 0, padx =100, pady = 5.5)
# textBox5.config(state='normal')

# textBox6 = tk.Text(frame5, width = 18, height = 1, wrap = "word")
# textBox6.grid(row = 5, column = 0, padx =100, pady = 5.5)
# textBox6.config(state='normal')


# Book Infos and Recommendations
plot = False
def get_book():
    global plot, canvas, canvas2
    value = entry.get()
    if len(value) > 0:

        delete_textboxes()

        if plot == True:
            frame6.winfo_children()[0].destroy()
            frame7.winfo_children()[0].destroy()

        true_book = find_book(value)
        canvas = FigureCanvasTkAgg(plot_book(true_book), master = frame6)
        canvas.draw()
        canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = 1)
        plot = True
        canvas2 = FigureCanvasTkAgg(plot_5_books(content_based_recommender(true_book)), master = frame7)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = 1)

        textBox2.insert(tk.END, true_book)
        # label21 = tk.Label(window, text = true_book, font = 'Verdana 10',
        #               fg = 'black', wraplength = 500)
        # label21.place(x = 285, y = 95)

        author = df.loc[df['book_title'] == true_book, 'book_author'].values[0]
        textBox3.insert(tk.END, author)
        # label31 = tk.Label(window, text = author, font = 'Verdana 10', 
        #               fg = 'black', wraplength = 500)
        # label31.place(x = 290, y = 125)
        
        category = df.loc[df['book_title'] == true_book, 'Category'].values[0]
        textBox4.insert(tk.END, category)
        # label41 = tk.Label(window, text = category, font = 'Verdana 10',
        #               fg = 'black', wraplength = 500)
        # label41.place(x = 305, y = 155)

        year = df.loc[df['book_title'] == true_book, 'year_of_publication'].values[0]
        textBox5.insert(tk.END, int(year))
        # label51 = tk.Label(window, text = int(year), font = 'Verdana 10',
        #                   fg = 'black', wraplength = 50)
        # label51.place(x = 365, y = 185)

        lang = df.loc[df['book_title'] == true_book, 'Language'].values[0]
        textBox6.insert(tk.END, lang)
        # label61 = tk.Label(window, text = lang, font = 'Verdana 10',
        #                   fg = 'black', wraplength = 80)
        # label61.place(x = 310, y = 215)
        
        country = df.loc[df['book_title'] == true_book, 'country'].values[0]
        textBox7.insert(tk.END, country.upper())
        # label71= tk.Label(window, text = country.upper(), font = 'Verdana 10',
        #                   fg = 'black', wraplength = 60)
        # label71.place(x = 300, y = 245)
        
        summary= df.loc[df['book_title'] == true_book, 'Summary'].values[0]
        textBox.insert(tk.END, summary)
        
        # print(content_based_recommender(true_book))
        # print(frame6.winfo_children()[0].destroy())
        
# Search Button
button = tk.Button(window, text = 'Search Book',
                   fg = 'black',height = 1,
                   width = 15, command = get_book)

button.place(x = 755, y = 50)

# Entry
entry = tk.Entry(window,width = 115)
entry.insert(string = '', index = 0, ) 
entry.place(x = 40, y = 55)

label1 = tk.Label(window, text = 'Book Recommender',
                  font = 'Verdana 18', fg = 'black',
                  wraplength = 450)

label1.place(x = 330, y = 3)

window.mainloop()







