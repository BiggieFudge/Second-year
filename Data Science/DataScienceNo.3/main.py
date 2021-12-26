# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# === CELL TYPE: IMPORTS AND SETUP

import time      # for testing use only
import os         # for testing use only

import bs4
from bs4 import BeautifulSoup
import pandas as pd
import scipy as sc
import numpy as np


"""1.a. Load the Data

As mentioned above, we cached the web pages.

    You should refer to the cached HTML files, located in the data folder.

For every html file use load a Beautiful Soup object.
You will later use this object to scrape information from this web-page.

Don't forget that all the html files are cached and are located in a sub-folder called 'data'. Thus, to the given 'html_file_name' input parameter, you need to add a './data/' prefix, before loading the soap object. (i.e. use './data/'+html_file_name instead of html_file_name).
Instructions

method name: load_soup_object

The following is expected:
--- Complete the 'load_soup_object' function to create and return a soup object 
    for a given html file.

    Note that all the html files are cached and are located in a sub-folder called 'data'.
    Thus, to the given 'html_file_name' input parameter, you need to add a './data/' prefix, 
    before loading the soap object.

   (i.e. use './data/'+html_file_name instead of html_file_name).    
"""

# 1.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER

def load_soup_object(html_file_name):
    url ='https://www.numbeo.com/quality-of-life/rankings_by_country.jsp?title=2021'+ html_file_name
    content = open(url)
    r = BeautifulSoup(content)
    return r

"""1.b. Extract IMDb movie genres

In this sub-section you will scape a list of movie genres from the main html page.

You need to scrape the name of the movie genres (such as 'adventure', 'musical' and so on) and the link to the top rated movies by these genres.
For example, the link to the adventure page on this site is "adventure.html".

In this question you need to scrape all the genres that appear on the page, and their corresponding links, and return the results as a dataframe.
Below you can see a sample dataframe with 2 rows (obviously there are more links on the page):
	genre_name 	link_to_genre_page
1 	Adventure 	adventure.html
2 	Animation 	animation.html
Instructions

method name: scrape_movie_genre_links

The following is expected:
--- Complete the 'scrape_movie_genre_links' function to scrape the movie 
    genre information described above,
    from a soup object corresponding to a given 'html_file_name' file. 

    You need to return a dataframe with the following columns:
    'genre_name', 'link_to_genre_page'

    Each row in the dataframe, should contain the information for 
    these 2 columns (as described above).
"""


# 1.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER

def scrape_movie_genre_links(html_file_name):
    list = []
    content = open('./data/' + html_file_name)
    bs = BeautifulSoup(content)
    for t in bs.findAll("li"):
        list.append([t.a.text, './data/' + t.a.get('href')])

    df = pd.DataFrame(list, columns=['genre_name', 'link_to_genre_page'])
    return df

"""Instructions

method name: load_top_rated_movies_per_genre

The following is expected:
--- Complete the 'load_top_rated_movies_per_genre' function to scrape all the required 
    information for each of the top movies, as described above, for a specific genre, 
    given in the 'genre_url_address' parameter.

    You need to crawl to the next pages of the genre, using the link to the 'Next' page.
    You could expect a total number of between 1-5 pages (the 'n_pages' parameter) to scrape (including 
    'genre_url_address', the first page of the genre).

    Use the previous 'load_soup_object' method to get a soup object for each of the 
    top rated movies web pages.

    We recommend that you create a single soup using BeautifulSoups' append function
    (for additional information visit documentation at:
    https://www.crummy.com/software/BeautifulSoup/bs4/doc/#append).

    You need to return a dataframe with the following columns:
    'movie_name', 'release_year', 'genre_names', 'rating'
"""


# 2.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER

def load_top_rated_movies_per_genre(genre_url_address, n_pages):
    bs = [load_soup_object(genre_url_address)]
    page = 51
    list = []
    i = 0
    for i in range(n_pages):
        for t in bs[i].findAll("div", attrs={"class": "lister-item-content"}):
            if (n_pages > 0):
                name = t.a.text
                release_year = t.find("span", {"class": "lister-item-year text-muted unbold"}).text
                genre_names = t.find("span", {"class": "genre"}).text
                rating = t.find("div", {"class": "ratings-bar"}).strong.text
                list.append([name, release_year, genre_names, rating])
        bs.append(load_soup_object(genre_url_address[:-5] + "_start_" + str(page) + ".html"))
        page += 50

    df = pd.DataFrame(list, columns=['movie_name', 'release_year', 'genre_names', 'rating'])
    # print (df)
    return df


