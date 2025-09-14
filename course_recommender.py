import anthropic
import json
import os
from typing import List, Dict, Any, Optional
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CourseRecommendationSystem:
    def __init__(self, api_key: str):
        """Initialize the course recommendation system with Claude API key."""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"

        # Available departments with course counts
        self.available_departments = {
            "Aeronautics_and_Astronautics": 92,
            "Anthropology": 67,
            "Architecture": 116,
            "Athletics,_Physical_Education_and_Recreation": 10,
            "Biological_Engineering": 41,
            "Biology": 85,
            "Brain_and_Cognitive_Sciences": 99,
            "Chemical_Engineering": 57,
            "Chemistry": 43,
            "Civil_and_Environmental_Engineering": 105,
            "Comparative_Media_Studies_Writing": 71,
            "Concourse": 5,
            "Earth,_Atmospheric,_and_Planetary_Sciences": 111,
            "Economics": 85,
            "Edgerton_Center": 28,
            "Electrical_Engineering_and_Computer_Science": 298,
            "Engineering_Systems_Division": 66,
            "Experimental_Study_Group": 30,
            "Global_Studies_and_Languages": 121,
            "Health_Sciences_and_Technology": 72,
            "History": 91,
            "Institute_for_Data,_Systems,_and_Society": 20,
            "Linguistics_and_Philosophy": 85,
            "Literature": 128,
            "Materials_Science_and_Engineering": 92,
            "Mathematics": 213,
            "Mechanical_Engineering": 162,
            "Media_Arts_and_Sciences": 47,
            "Music_and_Theater_Arts": 69,
            "Nuclear_Science_and_Engineering": 53,
            "Others": 447,
            "Physics": 117,
            "Political_Science": 97,
            "Science,_Technology_and_Society": 70,
            "Sloan_School_of_Management": 218,
            "Special_Programs": 2,
            "Urban_Studies_and_Planning": 211,
            "Women's_and_Gender_Studies": 61
        }

    def _call_claude_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Make API call to Claude with retry logic and error handling."""
        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All API call attempts failed: {str(e)}")
                    return None
        return None

    def select_departments(self, advisor_description: str, conversation_transcript: str,
                           skill_levels: List[List[str]]) -> List[str]:
        """
        Function 1: Analyze student profile and select the 3 most relevant departments.

        Args:
            advisor_description: Academic advisor's assessment of student
            conversation_transcript: Q&A dialogue between advisor and student
            skill_levels: Array of [skill_name, skill_level] pairs

        Returns:
            List of selected department names (1-3 departments)
        """
        # Format skill levels for the prompt
        skills_text = "\n".join([f"- {skill[0]}: {skill[1]}" for skill in skill_levels])

        # Create departments list for prompt
        departments_text = "\n".join(
            [f"- {dept}: {count} courses" for dept, count in self.available_departments.items()])

        prompt = f"""You are an expert academic advisor analyzing student profiles to recommend university departments from MIT.

STUDENT PROFILE:
Advisor Assessment: {advisor_description}

Conversation Context: {conversation_transcript}

Current Skills and Levels:
{skills_text}

AVAILABLE DEPARTMENTS:
{departments_text}

CRITICAL SELECTION RULES:
1. Select 1-3 departments maximum (only high-confidence matches)
2. Quality over quantity - don't force 3 if fewer are appropriate
3. MANDATORY: If student has low/beginner STEM skills but wants STEM field, MUST include foundational departments:
   - Mathematics (for mathematical foundations)
   - Physics (for science foundations) 
   - Electrical_Engineering_and_Computer_Science (for programming/engineering foundations)
4. Consider student's interests, current skill level, and learning goals
5. Only select departments you're highly confident will benefit the student

RESPONSE FORMAT:
Return ONLY a JSON array of department names, for example:
["Mathematics", "Physics", "Electrical_Engineering_and_Computer_Science"]

Analyze the student profile and select the most appropriate departments."""

        try:
            response = self._call_claude_api(prompt)
            if response:
                # Extract JSON from response
                import re
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if json_match:
                    departments = json.loads(json_match.group())
                    # Validate departments exist
                    valid_departments = [dept for dept in departments if dept in self.available_departments]
                    logger.info(f"Selected departments: {valid_departments}")
                    return valid_departments[:3]  # Ensure max 3 departments

            logger.error("Failed to parse department selection from Claude response")
            return []

        except Exception as e:
            logger.error(f"Error in select_departments: {str(e)}")
            return []

    def _load_department_courses(self, department: str) -> List[Dict]:
        """Load courses from department JSON file."""
        try:
            file_path = f"departments/{department}.json"
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    courses = json.load(f)
                    return courses
            else:
                logger.warning(f"Department file not found: {file_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading department {department}: {str(e)}")
            return []

    def select_courses_with_prerequisites(self, selected_departments: List[str], advisor_description: str,
                                          conversation_transcript: str, skill_levels: List[List[str]]) -> List[Dict]:
        """
        Function 2: Select specific courses from chosen departments and identify prerequisites.

        Args:
            selected_departments: Output from select_departments function
            advisor_description: Same as Function 1
            conversation_transcript: Same as Function 1
            skill_levels: Same as Function 1

        Returns:
            List of course objects with prerequisites
        """
        all_selected_courses = []

        # Format skill levels for prompt
        skills_text = "\n".join([f"- {skill[0]}: {skill[1]}" for skill in skill_levels])

        for department in selected_departments:
            # Load courses from department
            department_courses = self._load_department_courses(department)
            if not department_courses:
                continue

            # Prepare course list for Claude (title and short description)
            courses_text = "\n".join([
                f"- {course.get('title', 'No Title')}: {course.get('short_description', 'No Description')}"
                for course in department_courses[:50]  # Limit to avoid token limits
            ])

            prompt = f"""You are selecting specific courses from {department} department for a student based on their profile.

            STUDENT PROFILE:
            Advisor Assessment: {advisor_description}

            Conversation Context: {conversation_transcript}

            Current Skills and Levels:
            {skills_text}

            AVAILABLE COURSES IN {department}:
            {courses_text}

            SELECTION REQUIREMENTS:
            1. Select courses that closely match student interests and goals
            2. Consider skill level for difficulty appropriateness
            3. For each selected course, identify ALL necessary prerequisites based on student's current skill level
            4. CRITICAL: Prerequisites must be ACTUAL course titles from the available courses, not generic names
            5. Prerequisite Logic:
               - Beginner STEM students: Include foundational courses (Precalculus, Calculus, Basic Physics, etc.)
               - Intermediate students: Some foundational courses, can skip very basics
               - Advanced students: Direct access to advanced courses with minimal prerequisites
            6. Only select courses you're confident will benefit this specific student
            7. Quality over quantity - better to select fewer, more relevant courses

            RESPONSE FORMAT:
Return ONLY a valid JSON array. CRITICAL JSON RULES:
- NO line breaks anywhere in the JSON
- NO quotes inside descriptions 
- Keep descriptions under 500 characters
- Use simple punctuation only (periods, commas)
- Select as many courses per department as needed. Do not histate to add as many prerequisites as need. Add many and be happy. I need from you the comprehending guide for the user. With the full list of courses.

RESPONSE FORMAT:
            Return ONLY a valid JSON array. CRITICAL JSON RULES:
            - NO line breaks anywhere in the JSON
            - NO quotes inside descriptions 
            - Keep descriptions under 50 characters
            - Use simple punctuation only (periods, commas)
            - Select as many courses per department as needed

            Return format: JSON array of course objects with course_title, course_description, department, and prerequisites fields.

            IMPORTANT: Return ONLY the JSON array, nothing else. No explanations.

            Select appropriate courses with complete prerequisite chains for this student."""

            try:
                response = self._call_claude_api(prompt)
                if response:
                    import re

                    # Find the complete JSON array using bracket counting
                    start_idx = response.find('[')
                    if start_idx != -1:
                        bracket_count = 0
                        end_idx = start_idx

                        for i, char in enumerate(response[start_idx:], start_idx):
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_idx = i + 1
                                    break

                        json_str = response[start_idx:end_idx]

                        # Clean the JSON string
                        json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                        json_str = re.sub(r'\s+', ' ', json_str)  # Multiple spaces to single

                        try:
                            courses = json.loads(json_str)
                            if isinstance(courses, list):
                                all_selected_courses.extend(courses)
                                logger.info(f"Selected {len(courses)} courses from {department}")
                                if isinstance(courses, list):
                                    # Enrich courses with original descriptions from department_courses
                                    for course in courses:
                                        course_title = course.get('course_title', '')
                                        # Find original course in department_courses
                                        for original_course in department_courses:
                                            if original_course.get('title', '') == course_title:
                                                course['original_description'] = original_course.get(
                                                    'short_description', 'No description')
                                                break

                                    all_selected_courses.extend(courses)
                                    logger.info(f"Selected {len(courses)} courses from {department}")
                            else:
                                logger.error(f"Invalid JSON structure from {department}")
                        except json.JSONDecodeError as je:
                            logger.error(f"JSON decode error for {department}: {str(je)}")
                            logger.error(f"Problematic JSON: {json_str}")
                    else:
                        logger.error(f"No JSON array found in response for {department}")
                        logger.error(f"Full response: {response}")

            except Exception as e:
                logger.error(f"Error selecting courses from {department}: {str(e)}")



        logger.info(f"Total selected courses with prerequisites: {len(all_selected_courses)}")
        return all_selected_courses

    def create_learning_roadmap(self, courses_with_prereqs: List[Dict], student_profile: Dict = None) -> List:
        """
        Function 3: Create a graph representation from courses and prerequisites.

        Args:
            courses_with_prereqs: Output from select_courses_with_prerequisites
            student_profile: Not used, kept for compatibility

        Returns:
            List in format: [[vertices], [edges]]
            vertices: List of [course_name, original_course_description] pairs
            edges: List of [prerequisite_course, dependent_course] pairs
        """
        if not courses_with_prereqs:
            logger.error("No courses provided for graph creation")
            return [[], []]

        # Create vertices (nodes) - each course is a vertex
        vertices = []
        course_names = set()

        # Collect all course names first (including prerequisites)
        all_course_names = set()
        for course in courses_with_prereqs:
            course_name = course.get('course_title', 'Unknown')
            all_course_names.add(course_name)

            # Add prerequisite names too
            for prereq in course.get('prerequisites', []):
                all_course_names.add(prereq)

        # Create vertices list with original descriptions
        for course in courses_with_prereqs:
            course_name = course.get('course_title', 'Unknown')

            # Try to get original description from the course data
            # First try course_description, then short_description as fallback
            # Try to get original description from the course data
            original_desc = course.get('original_description', '')
            if not original_desc:
                original_desc = course.get('course_description', '')
            if not original_desc or original_desc == 'No description':
                original_desc = 'No description available'

            vertices.append([course_name, original_desc])
            course_names.add(course_name)

        # Add vertices for prerequisites that aren't in main course list
        for course in courses_with_prereqs:
            for prereq in course.get('prerequisites', []):
                if prereq not in course_names:
                    vertices.append([prereq, "Prerequisite course - description not available"])
                    course_names.add(prereq)

        # Create edges - from prerequisite to dependent course
        edges = []
        for course in courses_with_prereqs:
            course_name = course.get('course_title', 'Unknown')
            prerequisites = course.get('prerequisites', [])

            for prereq in prerequisites:
                # Edge format: [prerequisite_course, dependent_course]
                edges.append([prereq, course_name])

        logger.info(f"Created graph with {len(vertices)} vertices and {len(edges)} edges")
        return [vertices, edges]


def generate_course_roadmap(advisor_description: str, conversation_transcript: str,
                            skill_levels: List[List[str]]) -> List:
    """
    Main function to generate course roadmap from student profile.

    Args:
        api_key: Your Claude API key
        advisor_description: Academic advisor's assessment of the student
        conversation_transcript: Q&A dialogue between advisor and student
        skill_levels: Array of [skill_name, skill_level] pairs

    Returns:
        List in format: [[vertices], [edges]] where:
        - vertices: List of [course_name, course_description] pairs
        - edges: List of [prerequisite_course, dependent_course] pairs

    Example input format:
        api_key = "sk-ant-api03-..."
        advisor_description = "Student interested in AI/ML, software developer background, wants to transition to AI research"
        conversation_transcript = "Advisor: What interests you? Student: Neural networks and deep learning..."
        skill_levels = [["Mathematics", "Beginner"], ["Programming", "Intermediate"], ["Statistics", "Beginner"]]
    """
    try:
        api_key = "my-key"
        # Initialize the recommendation system
        recommender = CourseRecommendationSystem(api_key)

        # Step 1: Select departments
        departments = recommender.select_departments(advisor_description, conversation_transcript, skill_levels)

        if not departments:
            logger.warning("No departments selected, returning empty graph")
            return [[], []]

        # Step 2: Select courses with prerequisites
        courses = recommender.select_courses_with_prerequisites(
            departments, advisor_description, conversation_transcript, skill_levels
        )

        if not courses:
            logger.warning("No courses selected, returning empty graph")
            return [[], []]

        # Step 3: Create course graph
        student_profile = {
            "interests": "Based on advisor assessment",
            "background": "From conversation context",
            "goal": "Derived from student profile"
        }
        graph = recommender.create_learning_roadmap(courses, student_profile)

        logger.info(f"Successfully generated roadmap with {len(graph[0])} courses and {len(graph[1])} dependencies")
        return graph

    except Exception as e:
        logger.error(f"Error generating course roadmap: {str(e)}")
        return [[], []]








def main():
    """Example usage of the Course Recommendation System."""
    # Initialize system with your API key
    api_key = "my-key"
    recommender = CourseRecommendationSystem(api_key)

    # Example student data
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

    print("=== Course Recommendation System Demo ===\n")

    # Step 1: Select departments
    print("Step 1: Selecting relevant departments...")
    departments = recommender.select_departments(advisor_description, conversation_transcript, skill_levels)
    print(f"Selected departments: {departments}\n")

    if not departments:
        print("No departments selected. Exiting.")
        return

    # Step 2: Select courses with prerequisites
    print("Step 2: Selecting courses with prerequisites...")
    courses = recommender.select_courses_with_prerequisites(
        departments, advisor_description, conversation_transcript, skill_levels
    )
    print(f"Selected {len(courses)} courses")
    for course in courses:
        print(f"- {course.get('course_title', 'Unknown')} ({course.get('department', 'Unknown')})")
        if course.get('prerequisites'):
            print(f"  Prerequisites: {', '.join(course['prerequisites'])}")
    print()

    # Step 3: Create course graph
    # Step 3: Create course graph
    print("Step 3: Creating course graph...")
    student_profile = {
        "interests": "AI/ML",
        "background": "Software Developer",
        "goal": "AI Engineer/Researcher"
    }
    graph = recommender.create_learning_roadmap(courses, student_profile)

    vertices, edges = graph
    print(f"\n=== COURSE GRAPH ===")
    print(f"Total vertices (courses): {len(vertices)}")
    print(f"Total edges (dependencies): {len(edges)}")

    print("\nVertices (first 10 courses):")
    for i, vertex in enumerate(vertices[:10]):
        print(f"{i + 1}. {vertex[0]}")
        print(f"   Description: {vertex[1][:100]}...")
    if len(vertices) > 10:
        print(f"... and {len(vertices) - 10} more courses")

    print("\nEdges (first 15 dependencies):")
    for i, edge in enumerate(edges[:15]):
        print(f"{i + 1}. {edge[0]} → {edge[1]}")
    if len(edges) > 15:
        print(f"... and {len(edges) - 15} more dependencies")

    print(f"\nFinal graph structure: [[{len(vertices)} vertices], [{len(edges)} edges]]")

    print(graph)


if __name__ == "__main__":
    main()