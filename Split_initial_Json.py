import json
import os
from collections import defaultdict


def split_courses_by_department(input_file_path, output_directory="departments"):
    """
    Разделяет базу данных курсов на отдельные JSON файлы по департаментам

    Args:
        input_file_path (str): Путь к файлу all_course.json
        output_directory (str): Директория для сохранения файлов департаментов
    """

    # Создаем директорию для выходных файлов, если её нет
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Загружаем данные из исходного файла
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            courses_data = json.load(file)
    except FileNotFoundError:
        print(f"Файл {input_file_path} не найден!")
        return
    except json.JSONDecodeError:
        print(f"Ошибка при чтении JSON из файла {input_file_path}")
        return

    # Создаем словарь для группировки курсов по департаментам
    departments = defaultdict(list)

    # Обрабатываем каждый курс
    for course in courses_data:
        # Получаем названия департаментов
        department_names = course.get('department_name', [])

        # Если департаментов нет или список пустой, добавляем в "Others"
        if not department_names:
            departments['Others'].append(course)
        else:
            # Добавляем курс в каждый департамент, к которому он принадлежит
            for dept_name in department_names:
                # Очищаем название департамента и делаем безопасным для имени файла
                clean_dept_name = clean_filename(dept_name)
                departments[clean_dept_name].append(course)

    # Сохраняем каждый департамент в отдельный JSON файл
    for dept_name, dept_courses in departments.items():
        output_file = os.path.join(output_directory, f"{dept_name}.json")

        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(dept_courses, file, ensure_ascii=False, indent=2)

            print(f"Создан файл {output_file} с {len(dept_courses)} курсами")

        except Exception as e:
            print(f"Ошибка при сохранении файла {output_file}: {e}")

    # Выводим статистику
    print(f"\nСтатистика:")
    print(f"Всего департаментов: {len(departments)}")
    print(f"Всего курсов обработано: {len(courses_data)}")

    # Выводим информацию о каждом департаменте
    for dept_name, dept_courses in sorted(departments.items()):
        print(f"- {dept_name}: {len(dept_courses)} курсов")


def clean_filename(filename):
    """
    Очищает название файла от недопустимых символов

    Args:
        filename (str): Исходное название

    Returns:
        str: Очищенное название, безопасное для имени файла
    """
    # Заменяем недопустимые символы на подчеркивание
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Убираем лишние пробелы и заменяем их на подчеркивания
    filename = filename.strip().replace(' ', '_')

    # Удаляем множественные подчеркивания
    while '__' in filename:
        filename = filename.replace('__', '_')

    return filename


def get_all_departments(input_file_path):
    """
    Возвращает список всех уникальных департаментов в базе данных

    Args:
        input_file_path (str): Путь к файлу all_course.json

    Returns:
        list: Список уникальных названий департаментов
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            courses_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Ошибка при чтении файла: {e}")
        return []

    departments = set()

    for course in courses_data:
        department_names = course.get('department_name', [])
        if department_names:
            departments.update(department_names)

    return sorted(list(departments))


# Основной код для выполнения
if __name__ == "__main__":
    # Укажите путь к вашему файлу all_course.json
    input_file = "all_courses.json"  # Измените на правильный путь к вашему файлу

    print("Начинаю обработку базы данных курсов...")

    # Сначала покажем все доступные департаменты
    print("Найденные департаменты:")
    departments_list = get_all_departments(input_file)
    for i, dept in enumerate(departments_list, 1):
        print(f"{i:2d}. {dept}")

    print(f"\nВсего уникальных департаментов: {len(departments_list)}")
    print("-" * 50)

    # Разделяем курсы по департаментам
    split_courses_by_department(input_file, "departments")

    print("\nОбработка завершена!")