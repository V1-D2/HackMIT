import course_recommender
from course_recommender import generate_course_roadmap

# Ваши данные
api_key = "sk-ant-api03-..."
advisor_description = """
    Student shows strong interest in artificial intelligence and machine learning. 
    They want to build AI systems and understand deep learning algorithms. 
    Currently working as a software developer but lacks formal CS education. 
    Highly motivated to transition into AI research or engineering roles.
    """
conversation_transcript = """
    Advisor: What specifically interests you about AI?
    Student: I'm fascinated by neural networks and want to build systems that can understand language and images.

    Advisor: Do you have experience with mathematics and statistics?
    Student: I know basic programming but my math is rusty from high school.

    Advisor: What's your ultimate career goal?
    Student: I want to work at an AI company or do research in machine learning.
    """
skill_levels = [
        ["Mathematics", "Beginner"],
        ["Programming", "Intermediate"],
        ["Statistics", "Beginner"],
        ["Machine Learning", "Beginner"]
    ]

# Получить граф курсов
graph = generate_course_roadmap(advisor_description, conversation_transcript, skill_levels)
vertices, edges = graph
print(vertices)
print(graph)

# vertices содержит: [["Course Name", "Course Description"], ...]
# edges содержит: [["Prerequisite Course", "Dependent Course"], ...]