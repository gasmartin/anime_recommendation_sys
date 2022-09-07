import os
import time

import pandas as pd
import streamlit as st
from bing_image_downloader import downloader
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.title('Anime Recommendation System (σ‾▿‾)-σ')

# Aluno:    William do Vale Cavalcante Freitas
#           Gabriel Sena San Martin

if 'anime_df' not in st.session_state:
    st.session_state.anime_df = pd.read_csv('anime.csv').dropna()


if 'cosine_sim_df' not in st.session_state:
    tfidf = TfidfVectorizer(stop_words='english')
    st.session_state.cosine_similarity_df = pd.DataFrame(
        cosine_similarity(tfidf.fit_transform(st.session_state.anime_df['genre'])), 
        index=st.session_state.anime_df.name,
        columns=st.session_state.anime_df.name
    )
    st.session_state.cosine_similarity_df.sample(5, axis=1).round(2)


# @st.cache(suppress_st_warning=True)
# def load_data():
#     anime = pd.read_csv('anime.csv').dropna()
#     tfidf = TfidfVectorizer(stop_words='english')
#     cosine_sim_df = pd.DataFrame(cosine_similarity(tfidf.fit_transform(anime['genre'])), index=anime.name,
#                                  columns=anime.name)
#     cosine_sim_df.sample(5, axis=1).round(2)
#     return cosine_sim_df, anime


def paginator(label, items, items_per_page=10, on_sidebar=True):
    """Lets the user paginate a set of items.
    Parameters
    ----------
    label : str
        The label to display over the pagination widget.
    items : Iterator[Any]
        The items to display in the paginator.
    items_per_page: int
        The number of items to display per page.
    on_sidebar: bool
        Whether to display the paginator widget on the sidebar.

    Returns
    -------
    Iterator[Tuple[int, Any]]
        An iterator over *only the items on that page*, including
        the item's index.
    Example
    -------
    """

    # Figure out where to display the paginator
    if on_sidebar:
        location = st.sidebar.empty()
    else:
        location = st.empty()

    # Display a pagination selectbox in the specified location.
    items = list(items)
    n_pages = len(items)
    n_pages = (len(items) - 1) // items_per_page + 1
    page_format_func = lambda i: "Page %s" % i
    page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

    # Iterate over the items in the page to let the user display them.
    min_index = page_number * items_per_page
    max_index = min_index + items_per_page
    import itertools
    return itertools.islice(enumerate(items), min_index, max_index)


def genre_recommendations(anime_name, similarity_matrix, items, k=10):
    anime_loc = similarity_matrix.loc[:, anime_name].to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_matrix.columns[anime_loc[-1:-(k + 2):-1]]
    closest = closest.drop(anime_name, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)


def download_images(selected_recommendation_names):
    for name in selected_recommendation_names:
        name = name.replace('/', ' ')
        list_dir = os.listdir(os.path.join(downloads, name))
        if not list_dir:
            downloader.download(name, limit=1, output_dir='downloads',
                                adult_filter_off=True, force_replace=False,
                                timeout=60, verbose=True)


def suggest_anime(anime_name, similarity_threshold=75):
    suggestions = []

    for i, r in st.session_state.anime_df.iterrows():
        score = fuzz.token_sort_ratio(anime_name, r['name'])
        if score >= similarity_threshold:
            suggestions.append([r['name'], score])

    suggestions.sort(key = lambda suggestions: suggestions[1])
    try:
        return suggestions[0][0]
    except IndexError:
        pass


# st.text(open("summary.txt", 'r').read())
# cosine_similarity_df, anime_df = load_data()


str_new_user = st.text_area('Enter the anime to be analyzed')
downloads = os.path.join(os.path.dirname(__file__), 'downloads')

if not os.path.exists(downloads):
    os.makedirs(downloads)

downloads_path = [os.path.join(downloads, p) for p in os.listdir(os.path.join(os.path.dirname(__file__), 'downloads'))]

for i, r in st.session_state.anime_df.iterrows():
    name = r['name'].replace('/', ' ')
    path = os.path.join(downloads, name)
    if not os.path.isdir(path):
        os.mkdir(path)

if str_new_user and str_new_user != '':
    try:
        with st.spinner('Loading recommendations...'):
            recommendations = genre_recommendations(str_new_user, st.session_state.cosine_similarity_df, st.session_state.anime_df[['name', 'genre']])

            recommendations = {k: list(v.values()) for k, v in recommendations.to_dict().items()}

            recommendation_names = recommendations['name'][:4]
            recommendation_genres = recommendations['genre'][:4]
            download_images(recommendation_names)

        st.subheader('Anime Recommendations')

        print('images downloaded')

        selected_images = []
        for name in recommendation_names:
            for folder in downloads_path:
                if name.replace('/', ' ') in folder:
                    for i in os.listdir(folder):
                        selected_images.append(os.path.join(folder, i))

        max_attempts = 5
        while max_attempts:
            try:
                print(selected_images)
                image_iterator = paginator("Select a sunset page", selected_images)
                indices_on_page, images_on_page = map(list, zip(*image_iterator))
                st.image(images_on_page, width=160, caption=[f"Name: {n} | Genre: {g}" for n, g in zip(recommendation_names,
                                                                                                    recommendation_genres)])
            except Exception as e:
                print(f'Script error: {e}')
                max_attempts -= 1
                time.sleep(5)
    except KeyError:
        suggestion = suggest_anime(str_new_user)
        message = f'Could not find anime "{str_new_user}".'
        if suggestion:
            message += f' Did you mean "{suggestion}"?'
        st.warning(message)
