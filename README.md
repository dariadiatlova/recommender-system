# Тестовое задание на летнюю стажировку в комаду CoreML Вконтакте


Задание: на датасете [MovieLens20M](https://www.kaggle.com/grouplens/movielens-20m-dataset/code)
сравнить два подхода к построению рекомендации:

- коллаборативный: используя только рейтинги. Например SVD-like алгоритмы, ALS, Implicit-ALS.

- коллаборативный + контентный: используя рейтинги и всю дополнительную информацию о фильмах, имеющуюся в датасете. Например LightFM.

Задачи:
1. Выбрать метрику и обосновать выбор.
2. Придумать и обосновать способ разбиения данных на обучение и валидацию.
3. Обратить внимание на сходимость обучения и настройку важных гиперпараметров моделей.
4. Оценить статистическую значимость результатов.

___

## Коллаборативный подход:
0. Для начала почистим данные. Скрипт [`filter_data.py`](src/preprocessing/filter_data.py)
   считает, сколько составляет 1% от всех пользователей и удаляет из датасета
   фильмы с меньшим числом оценок. Размер исходного датасета после филтрации сократился на 15%.
1. Для того, чтобы грамотно выбрать метрику необходимо уточнить задачу, которую мы решаем. Пусть мы хотим, чтобы наша рекомендательная система на этапе `evaluation` выдавала каждому пользователю N рекомендаций фильмов. Переведем пятибальную рейтинговую шкалу в бинарную, в которой оценки 4 и 5 будут соответсвовать метке `1` – фильм понравился, а остальные оценки метке `0` – фильм не понравился. Тогда нам подойдет метрика `MAP` – так как качество нашей модели характеризует то, на сколько релевантные оценки в среднем мы выдаем нашим пользователям. 
2. Для разбиения на тренировочный и тестовый датасет воспользуемся стратегией [`Temporal Global`](https://arxiv.org/pdf/2007.13237.pdf).
   Разобьем датасет – файл `raiting.csv` на тренировочный и валидационный согласно **выбранной дате** так, что в тренировочный датасет попадут все
   пользовательские действия, которые были совершены до выбранной даты, а в тестовый – оставшиеся.
   Разбиение файла `raiting.csv` осуществляется с помощью запуска скрипта [`split_dataset.py`](src/preprocessing/split_dataset.py)
   с командной строки. 
   Размер тестового датасета определяется случайным образом, но гарантировано попадает в заданный диапазон значений.
   Параметры разбиения заданы в [`util.py`](src/common/util.py). В нашем случае в тестовый датасет попало 30% данных.
