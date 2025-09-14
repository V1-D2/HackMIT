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
Return ONLY a JSON array of course objects:
[
  {{
    "course_title": "Actual Course Title from List",
    "course_description": "Course description",
    "department": "{department}",
    "prerequisites": ["Prerequisite Course 1", "Prerequisite Course 2"]
  }}
]

Select appropriate courses with complete prerequisite chains for this student."""

            try:
                response = self._call_claude_api(prompt)
                if response:
                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                    if json_match:
                        courses = json.loads(json_match.group())
                        all_selected_courses.extend(courses)
                        logger.info(f"Selected {len(courses)} courses from {department}")

            except Exception as e:
                logger.error(f"Error selecting courses from {department}: {str(e)}")

        logger.info(f"Total selected courses with prerequisites: {len(all_selected_courses)}")
        return all_selected_courses

    def create_learning_roadmap(self, courses_with_prereqs: List[Dict], student_profile: Dict = None) -> Dict:
        """
        Function 3: Create a structured learning path with proper course sequencing.

        Args:
            courses_with_prereqs: Output from select_courses_with_prerequisites
            student_profile: Additional student context (optional)

        Returns:
            Structured roadmap dictionary
        """
        if not courses_with_prereqs:
            return {"error": "No courses provided for roadmap creation"}

        # Prepare courses data for Claude
        courses_text = ""
        for i, course in enumerate(courses_with_prereqs):
            prereqs = ", ".join(course.get('prerequisites', []))
            courses_text += f"{i + 1}. Course: {course.get('course_title', 'Unknown')}\n"
            courses_text += f"   Department: {course.get('department', 'Unknown')}\n"
            courses_text += f"   Description: {course.get('course_description', 'No description')}\n"
            courses_text += f"   Prerequisites: {prereqs if prereqs else 'None'}\n\n"

        profile_text = ""
        if student_profile:
            profile_text = f"Student Profile Context: {student_profile}"

        prompt = f"""Create a structured learning roadmap from the selected courses and their prerequisites.

{profile_text}

COURSES WITH PREREQUISITES:
{courses_text}

ROADMAP REQUIREMENTS:
1. Analyze all courses and their prerequisite relationships
2. Create a level-based learning progression (Foundation → Intermediate → Advanced)
3. Ensure proper course sequencing - prerequisites must come before dependent courses
4. Group courses into logical learning levels
5. Estimate realistic timeframes for each level
6. Provide clear learning path summary

RESPONSE FORMAT:
Return ONLY a JSON object with this exact structure:
{{
  "roadmap": {{
    "levels": [
      {{
        "level": 1,
        "level_name": "Foundation",
        "courses": [
          {{
            "course_title": "Course Name",
            "description": "Course description",
            "department": "Department Name",
            "estimated_duration": "X weeks"
          }}
        ]
      }},
      {{
        "level": 2,
        "level_name": "Intermediate",
        "courses": [...]
      }},
      {{
        "level": 3,
        "level_name": "Advanced", 
        "courses": [...]
      }}
    ]
  }},
  "total_estimated_time": "X months/years",
  "learning_path_summary": "Comprehensive description of the complete learning journey and how courses build upon each other"
}}

Create an optimal learning sequence with proper dependencies."""

        try:
            response = self._call_claude_api(prompt)
            if response:
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    roadmap = json.loads(json_match.group())
                    logger.info("Successfully created learning roadmap")
                    return roadmap

            logger.error("Failed to parse roadmap from Claude response")
            return {"error": "Failed to create roadmap"}

        except Exception as e:
            logger.error(f"Error creating learning roadmap: {str(e)}")
            return {"error": f"Roadmap creation failed: {str(e)}"}


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

    # Step 3: Create learning roadmap
    print("Step 3: Creating learning roadmap...")
    student_profile = {
        "interests": "AI/ML",
        "background": "Software Developer",
        "goal": "AI Engineer/Researcher"
    }
    roadmap = recommender.create_learning_roadmap(courses, student_profile)

    if "error" in roadmap:
        print(f"Error creating roadmap: {roadmap['error']}")
        return

    # Display roadmap
    print("=== LEARNING ROADMAP ===")
    print(f"Total Estimated Time: {roadmap.get('total_estimated_time', 'Unknown')}")
    print(f"Summary: {roadmap.get('learning_path_summary', 'No summary')}\n")

    for level in roadmap.get('roadmap', {}).get('levels', []):
        print(f"LEVEL {level.get('level', '?')}: {level.get('level_name', 'Unknown')}")
        for course in level.get('courses', []):
            print(
                f"  • {course.get('course_title', 'Unknown')} ({course.get('estimated_duration', 'Unknown duration')})")
            print(f"    {course.get('description', 'No description')}")
            print(f"    Department: {course.get('department', 'Unknown')}")
        print()


if __name__ == "__main__":
    main()