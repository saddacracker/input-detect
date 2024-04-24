import cv2 as cv
import numpy as np
import os

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

def adjust_textfield_count(template_types):
  # For each template type, check if it is a textField
  for templateType in template_types:
    if template_types[templateType] > 0:
      if templateType == 'textField':
        # update the count to be /2 since a textfield is made up of two pieces
        template_types[templateType] = int(template_types[templateType] / 2)
  return template_types
 
def compare_templates_to_screenshot(templates, img_gray, img_screenshot, current_screenshot):
    template_types = {}

    for key in templates:
        current_template = templates[key]
        template = cv.imread(current_template['template'], cv.IMREAD_GRAYSCALE)
        assert template is not None, "file could not be read, check with os.path.exists()"
        w, h = template.shape[::-1]

        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= current_template['threshold'])

        count_for_template = len(list(zip(*loc[::-1])))
        for pt in zip(*loc[::-1]):
            cv.rectangle(img_screenshot, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

        if count_for_template > 0:
            template_types[current_template['type']] = template_types.get(current_template['type'], 0) + count_for_template

        cv.imwrite(f'output/res_{current_screenshot}', img_screenshot)

    return template_types

def process_screenshots(screenshot_dir, templates):
    screenshot_files = [f for f in os.listdir(screenshot_dir) if f.endswith('.png')]

    for screenshot_file in screenshot_files:
        img_screenshot = cv.imread(os.path.join(screenshot_dir, screenshot_file))
        img_gray = cv.cvtColor(img_screenshot, cv.COLOR_BGR2GRAY)

        template_types = compare_templates_to_screenshot(templates, img_gray, img_screenshot, screenshot_file)

        adjusted_template_types = adjust_textfield_count(template_types)
        print(f"Screenshot: {screenshot_file}, Template Types: {adjusted_template_types}")


screenshot_dir = 'screenshots'
templates = {
 'singleDigitEmpty': 
 {'template': 'templates/images/singledigit-empty.png', 
  'threshold': 0.9,
  'type': 'singleDigit'
 },
 'singleDigitFilled': 
 {'template': 'templates/images/singledigit-filled.png', 
  'threshold': 0.9,
  'type': 'singleDigit'
 },
 'singleDigitSelected': 
 {'template': 'templates/images/singledigit-selected.png', 
  'threshold': 0.9,
  'type': 'singleDigit'
 },
 'textfieldStartSelected': 
 {'template': 'templates/images/textfieldstart-selected.png', 
  'threshold': 0.9,
  'type': 'textField'
 },
 'textfieldEndSelected': 
 {'template': 'templates/images/textfieldend-selected.png', 
  'threshold': 0.9,
  'type': 'textField'
 },
}
process_screenshots(screenshot_dir, templates)



