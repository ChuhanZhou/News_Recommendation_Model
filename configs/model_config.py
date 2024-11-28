import torch

config = {
    'category_label_num': 3000,#[2,2975] total:294
    'sentiment_label_dict': {'Negative':0,'Neutral':1,'Positive':2},
    'article_type_dict': {
        'article_default': 0,
        'article_webtv': 1,
        'article_page_nine_girl': 2,
        'article_questions_and_answers': 3,
        'article_feature': 4,
        'article_opinionen': 5,
        'article_native': 6,
        'article_scribblelive': 7,
        'article_fullscreen_gallery': 8,
        'article_editorial_production': 9,
        'article_standard_feature': 10,
        'article_native_feature': 11,
        'article_accordion': 12,
        'article_video_standalone': 13,
        'article_image_gallery': 14,
        'article_timeline': 15},
    'total_views_norm':1e7,
    'total_read_time_norm':1e9,
    'pca_vector':64,
    'subcategory_max_num':5,
    'history_max_num': 200,
    'inview_max_num':15,#90%:20, 80%:15, 75%:13 , 70%:12
}
