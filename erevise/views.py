# import os
# import re
# import json
# import pandas as pd
# import random
# from django.http import JsonResponse
# # os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'epl.settings')
# from django.http import HttpResponse
# from django.shortcuts import render, redirect
# from django.contrib import messages
# from database.models import User, Prompt, Essay, Feedback, Process, Classroom
# from .forms import UploadFileForm, UsernameForm
# from django.utils import timezone
# from .models import ScoreEssay, ScoreRevision
# from django.db.models import Q
# from bs4 import BeautifulSoup
# from django.contrib.auth import login as auth_login
# from django.contrib.auth.forms import UserCreationForm
#
# from django.shortcuts import render, redirect
# from django.contrib.auth import authenticate, login
# from .forms import UsersLoginForm
#
# from django.shortcuts import render, redirect
# from django.contrib.auth.forms import UserCreationForm
# from django.contrib.auth.models import User, Permission, Group
# from django.contrib.auth import authenticate, login
# from django.contrib.auth.forms import AuthenticationForm
# from django.contrib.auth import logout
# from .forms import EssayForm, UsersLoginForm, RevisionForm, CustomUserCreationForm
# from django.shortcuts import HttpResponseRedirect
# from django.conf import settings

import os
import json
import pandas as pd
from django.http import JsonResponse
from django.http import HttpResponse
from django.contrib import messages
from database.models import User, Prompt, Essay, Feedback, Process, Classroom, ProcessRevision
from .forms import UploadFileForm, UsernameForm
from django.utils import timezone
from .models import ScoreEssay, ScoreRevision
from django.db.models import Q
from bs4 import BeautifulSoup
from django.shortcuts import render, redirect
from django.contrib.auth.models import User, Permission, Group
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from .forms import EssayForm, UsersLoginForm, RevisionForm, CustomUserCreationForm
from django.conf import settings

def signup(request):
    if request.user.is_authenticated:
        return redirect('/')
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password, is_staff=True)
            login(request, user)
            return redirect('/')
        else:
            return render(request, 'signup.html', {'form': form})
    else:
        form = CustomUserCreationForm()
        return render(request, 'signup.html', {'form': form})


def signin(request):
    if request.user.is_authenticated:
        return render(request, 'index.html')
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('/')  # profile
        else:
            msg = 'Error Login'
            form = UsersLoginForm(request.POST)
            return render(request, 'login.html', {'form': form, 'msg': msg})
    else:
        form = UsersLoginForm()
        return render(request, 'login.html', {'form': form})



def signout(request):
    logout(request)
    return redirect('/')

def get_prompt(prompt_name="MVP"):
    init_prompt_text = Prompt.objects.filter(prompt_name=prompt_name).values_list("prompt_content")[0][0]
    init_prompt_text = init_prompt_text.replace("\n", " ")
    init_prompt_text = " ".join([token for token in init_prompt_text.split(" ")])
    init_prompt_text = init_prompt_text.strip()
    return init_prompt_text

def get_feedback(prompt_name, level, npe, spc):
    feedback = Feedback.objects.filter(prompt_name=prompt_name, level=level).values("title", "content")
    title = f'<h5>{feedback[0]["title"]}</h5>'
    content = feedback[0]["content"]
    # content = re.sub(re.compile('<.*?>'), '', content)
    # content_list = content.split("\n")
    # content_list = [c.strip() for c in content_list if len(c)>0]
    # result = {"title":title, "content":content}
    result = title + content
    return result

def get_evidence_feedback(prompt_name, level):
    if str(level) == "1":
        if prompt_name.lower() == "mvp":
            bullet1 = '<li style="padding-left: 0px;">Adding more evidence would make your argument even more convincing.</li>'
            bullet2 = '<li style="padding-left: 0px;">Reread the highlighted portions of the article to choose more evidence.</li>'
            message = bullet1 + bullet2
        else:
            bullet1 = '<li style="padding-left: 0px;">Adding more evidence would make your argument even more convincing.</li>'
            bullet2 = '<li style="padding-left: 0px;">If you think that that space exploration should be funded, reread the green highlighted portions of the article to choose more evidence to add in your essay.</li>'
            bullet3 = '<li style="padding-left: 0px;">If you think that space exploration should not be funded reread the pink highlighted portions of the article to choose more evidence to add in your essay.</li>'
            message = bullet1 + bullet2 + bullet3
    elif str(level) == "2":
        bullet1 = '<li style="padding-left: 0px;">Adding more details will help your reader better understand your ideas. This will make your argument even more convincing.</li>'
        bullet2 = '<li style="padding-left: 0px;">When you revise your essay, make sure to add more details for each piece of evidence you use.</li>'
        message = bullet1 + bullet2
    else:
        bullet1 = '<li style="padding-left: 0px;">Having evidence is important, but you need to help your reader understand how the evidence you chose supports your argument.</li>'
        bullet2 = '<li style="padding-left: 0px;">When you revise your essay, focus on explaining how each piece of evidence you used connects to your idea.</li>'
        bullet3 = '<li style="padding-left: 0px;">Give a detailed and clear explanation of how the evidence supports your argument.</li>'
        bullet4 = '<li style="padding-left: 0px;">Tie the evidence not only to the point you are making within a paragraph, but to your overall argument.</li>'
        message = bullet1 + bullet2 + bullet3 + bullet4
    return message

def get_revision_feedback(prompt_name, revision_level, evidence_level):
    if evidence_level == 1:
        evidence_message = get_evidence_feedback(prompt_name, evidence_level)
        if revision_level == 1.0:
            bullet1 = '<li style="padding-left: 0px;">When writers revise, they generally add more content. This often makes their essays longer</li>'
            bullet2 = '<li style="padding-left: 0px;">This time when you revise your essay, focus on adding more evidence.</li>'
            messages = bullet1 + bullet2 + evidence_message
        elif revision_level == 1.1:
            bullet1 = '<li style="padding-left: 0px;">When you revised your essay, it looks like you edited your writing to be clearer and easier for a reader to understand.</li>'
            bullet2 = '<li style="padding-left: 0px;">Revising is different from editing. When writers revise their essays, they generally add more content. This often makes their essays longer.</li>'
            bullet3 = '<li style="padding-left: 0px;">This time when you revise your essay, focus on adding more evidence.</li>'
            messages = bullet1 + bullet2 + bullet3 + evidence_message
        elif revision_level == 1.2:
            bullet1 = '<li style="padding-left: 0px;">When you revised your essay, it looks like you added in evidence that was very similar to the evidence you had included before.</li>'
            bullet2 = '<li style="padding-left: 0px;">When writers revise, they generally add new content to their essays.</li>'
            messages = bullet1 + bullet2 + evidence_message
        elif revision_level == 1.3:
            bullet1 = '<li style="padding-left: 0px;">When you revised your essay, it looks like you added more information about your thinking but did not include new information from the article.</li>'
            bullet2 = '<li style="padding-left: 0px;">When writers revise their text-based essays, they generally add new content from the text and delete content that is not based on the text.</li>'
            messages = bullet1 + bullet2 + evidence_message
        elif revision_level == 1.4:
            bullet1 = '<li style="padding-left: 0px;">When you revised your essay, it looks like you did not include new information from the article.</li>'
            bullet2 = '<li style="padding-left: 0px;">When writers revise their text-based essays, they generally add new content from the text and delete content that is not based on the text.</li>'
            messages = bullet1 + bullet2 + evidence_message
        elif revision_level == 1.5:
            bullet1 = '<li style="padding-left: 0px;">When you revised your essay, it looks like you followed the suggestion to add more evidence. Great job!</li>'
            bullet2 = '<li style="padding-left: 0px;">Paying attention to feedback is how people become stronger writers.</li>'

            bullet3 = '<li style="padding-left: 0px;">This time when you revise your essay, make sure to provide more details from the article for each piece of evidence you use.</li>'
            bullet4 = '<li style="padding-left: 0px;">Adding more details will help your reader better understand your ideas and this will make your argument even more convincing.</li>'
            messages = bullet1 + bullet2 + bullet3 + bullet4
        else:
            raise "Not implemented"

    elif evidence_level == 2:
        evidence_message = get_evidence_feedback(prompt_name, evidence_level)
        if revision_level == 2.0:
            bullet1 = '<li style="padding-left: 0px;">For example, if a student is writing an essay about water, the sentence, “People don’t have enough water” would not be specific enough. It would be better to write, “Only about 43% of the people in Mexico have access to water that is safe to drink.”</li>'
            bullet2 = '<li style="padding-left: 0px;">When writers revise, they generally add more content. This often makes their essays longer.</li>'
            bullet3 = '<li style="padding-left: 0px;">This time when you revise your essay, focus on adding more details to your evidence.</li>'
            messages = bullet2 + bullet3 + evidence_message + bullet1
        elif revision_level == 2.1:
            bullet1 = '<li style="padding-left: 0px;">For example, if a student is writing an essay about water, the sentence, “People don’t have enough water” would not be specific enough. It would be better to write, “Only about 43% of the people in Mexico have access to water that is safe to drink.”</li>'
            bullet2 = '<li style="padding-left: 0px;">When you revised your essay, it looks like you edited your writing to be clearer and easier for a reader to understand.</li>'
            bullet3 = '<li style="padding-left: 0px;">Revising is different from editing. When writers revise their essays, they generally add more content. This often makes their essays longer.</li>'
            bullet4 = '<li style="padding-left: 0px;">This time when you revise your essay, focus on adding more details to your evidence.</li>'
            messages = bullet2 + bullet3 + bullet4 + evidence_message + bullet1
        elif revision_level == 2.2:
            bullet1 = '<li style="padding-left: 0px;">For example, if a student is writing an essay about water, the sentence, “People don’t have enough water” would not be specific enough. It would be better to write, “Only about 43% of the people in Mexico have access to water that is safe to drink.”</li>'
            bullet2 = '<li style="padding-left: 0px;">When you revised your essay, it looks like you mentioned more evidence from the article.</li>'
            bullet3 = '<li style="padding-left: 0px;">When writers revise, they don’t just add more information. They also add more details to the information they already have in their essay. This often makes their essays longer.</li>'
            messages = bullet2 + bullet3  + evidence_message + bullet1
        elif revision_level == 2.3:
            bullet1 = '<li style="padding-left: 0px;">When you revised your essay, it looks like you added more information but did not include new details from the article.</li>'
            bullet2 = '<li style="padding-left: 0px;">When writers revise their text-based essays, they generally add new details from the text  and delete details that is not based on the text.</li>'
            messages = bullet1 + bullet2  + evidence_message
        elif revision_level == 2.4:
            bullet1 = '<li style="padding-left: 0px;">When you revised your essay, it looks like you followed the suggestion to add more evidence. Great job!</li>'
            bullet2 = '<li style="padding-left: 0px;">Paying attention to feedback is how people become stronger writers.</li>'

            bullet3 = '<li style="padding-left: 0px;">This time when you revise your essay, make sure to explain how each piece of evidence you used connects to your idea.</li>'
            bullet4 = '<li style="padding-left: 0px;">Give a detailed and clear explanation of how the evidence supports your argument.</li>'
            bullet5 = '<li style="padding-left: 0px;">Tie the evidence not only to the point you are making within a paragraph, but to your overall argument.</li>'
            bullet6 = '<li style="padding-left: 0px;">This will help your reader understand how the evidence you chose supports your argument.</li>'
            messages = bullet1 + bullet2 + bullet3 + bullet4 + bullet5 + bullet6
        else:
            raise "Not implemented"

    else:
        evidence_message = get_evidence_feedback(prompt_name, evidence_level)
        if evidence_level == 3.0:
            bullet1 = '<li style="padding-left: 0px;">For example, imagine that a student is writing an essay arguing that companies and factories need to stop polluting water because it is dangerous for people’s health. The student gives this evidence: “Only about 43% of the people in Mexico have access to water that is safe to drink.” A sentence linking the evidence to the claim might be, “Drinking unsafe water leads to diseases such as typhoid which can kill people.”</li>'
            bullet2 = '<li style="padding-left: 0px;">When writers revise their essays, they generally add more content. This often makes their essays longer.</li>'
            bullet3 = '<li style="padding-left: 0px;">This time when you revise your essay, focus on explaining how your evidence supports your claim.</li>'
            messages = bullet2 + bullet3  + evidence_message + bullet1
        elif revision_level == 3.1:
            bullet1 = '<li style="padding-left: 0px;">For example, imagine that a student is writing an essay arguing that companies and factories need to stop polluting water because it is dangerous for people’s health. The student gives this evidence: “Only about 43% of the people in Mexico have access to water that is safe to drink.” A sentence linking the evidence to the claim might be, “Drinking unsafe water leads to diseases such as typhoid which can kill people.”</li>'
            bullet2 = '<li style="padding-left: 0px;">When you revised your essay, it looks like you edited your writing to be clearer and easier for a reader to understand.</li>'
            bullet3 = '<li style="padding-left: 0px;">Revising is different from editing. When writers revise their essays, they generally add more content. This often makes their essays longer.</li>'
            bullet4 = '<li style="padding-left: 0px;">This time when you revise your essay, focus on explaining how your evidence supports your claim.</li>'
            messages = bullet2 + bullet3 + bullet4 + evidence_message + bullet1
        elif revision_level == 3.2:
            bullet1 = '<li style="padding-left: 0px;">For example, imagine that a student is writing an essay arguing that companies and factories need to stop polluting water because it is dangerous for people’s health. The student gives this evidence: “Only about 43% of the people in Mexico have access to water that is safe to drink.” A sentence linking the evidence to the claim might be, “Drinking unsafe water leads to diseases such as typhoid which can kill people.”</li>'
            bullet2 = '<li style="padding-left: 0px;">When you revised your essay, it looks like you may have focused on something other than explaining your evidence.</li>'
            bullet3 = '<li style="padding-left: 0px;">Revising the explanation or reasoning part of an essay is hard to do! When writers revise for this, they make sure that after providing a piece of evidence, they say something that connects it to their argument. The explanation should not just restate the evidence in different words.</li>'
            messages = bullet2 + bullet3 + evidence_message + bullet1
        elif revision_level == 3.3:
            bullet1 = '<li style="padding-left: 0px;">For example, imagine that a student is writing an essay arguing that companies and factories need to stop polluting water because it is dangerous for people’s health. The student gives this evidence: “Only about 43% of the people in Mexico have access to water that is safe to drink.” A sentence linking the evidence to the claim might be, “Drinking unsafe water leads to diseases such as typhoid which can kill people.”</li>'
            bullet2 = '<li style="padding-left: 0px;">When you revised your essay, it looks like you may have focused on something other than explaining your evidence.</li>'
            bullet3 = '<li style="padding-left: 0px;">Revising the explanation or reasoning part of an essay is hard to do! When writers revise for this, they make sure that after providing a piece of evidence, they say something that connects it to their argument. The explanation should not just restate the evidence in different words.</li>'
            messages = bullet2 + bullet3 + evidence_message + bullet1
        elif revision_level == 3.4:
            bullet1 = '<li style="padding-left: 0px;">When you revised your essay, it looks like you followed the suggestion to explain your evidence and how it connects to your claim. Great job!</li>'
            bullet2 = '<li style="padding-left: 0px;">Paying attention to feedback is how people become stronger writers.</li>'
            bullet3 = '<li style="padding-left: 0px;">This time when you revise your essay, try to use a counterclaim. This means bringing up a point that is against the side you are arguing for, but then explaining how that may actually support your side.</li>'
            bullet4 = '<li style="padding-left: 0px;">For example, imagine that a student is writing an essay arguing that there needs to be laws to make companies and factories pay for polluting water. A sentence that acknowledges and addresses an opposing view might be, “Some people argue that making companies pay will make things like clothes and cars more expensive. But paying doctors to heal diseases caused by water pollution is expensive also.”</li>'
            bullet5 = '<li style="padding-left: 0px;">Using a counterclaim makes your argument stronger. It shows that you have considered the opposite side but have stronger reasons for supporting your position.</li>'
            messages = bullet1 + bullet2 + bullet3 + bullet4 + bullet5
        else:
            raise "Not implemented"

    return messages




def parse_feedback(html):

    soup = BeautifulSoup(html, 'html.parser')

    # Extract title
    title = soup.find_all('h5')
    title_list = []
    for t in title:
        title_list.append(t.text)

    # Extract content
    content_list = []
    for ul in soup.find_all('ul'):
        content_list.append([li.text.strip() for li in ul.find_all('li')])

    # Create the result dictionary
    result = {
        'title': title_list,
        'content': content_list
    }
    return result

def get_essay_revision(prompt_name, revision_id):
    submited_essay = Essay.objects.filter(essay_revision=revision_id, prompt_name=prompt_name).order_by('-submitted_time')[:1].values("essay_content")
    essay_content = submited_essay[0]["essay_content"]
    essay_content = essay_content.replace("\n", " ")
    essay_content = " ".join([token for token in essay_content.split(" ")])
    essay_content = essay_content.strip()
    return essay_content

def get_essay_revision_v2(user_name, prompt_name, revision_id):
    submited_essay = Essay.objects.filter(user_name=user_name, essay_revision=revision_id, prompt_name=prompt_name).order_by('-submitted_time')[:1].values("essay_content")
    essay_content = submited_essay[0]["essay_content"]
    essay_content = essay_content.replace("\n", " ")
    essay_content = " ".join([token for token in essay_content.split(" ")])
    essay_content = essay_content.strip()
    return essay_content

def get_annotation_mvp(match_examples):
    missing_marker_mapper = {"HOSPITAL":[11], "MALARIA":[12,13], "FARMING":[16], "SCHOOL":[17]}
    markers = ["GENERAL", "GENERAL", "HOSPITAL", "MALARIA", "FARMING", "SCHOOL", "HOSPITAL", "GENERAL"]
    with open(os.path.join(settings.BASE_DIR, "templates/mvp_essay.html"), "r") as f:
        lines = f.readlines()

    marker_list = []
    for i in range(len(match_examples)):
        if len(match_examples[i]) > 0:
            marker = markers[i]
            marker_list.append(marker)

    full_list = set(["HOSPITAL", "MALARIA", "FARMING", "SCHOOL"])
    marker_list = set(marker_list)
    missing_list = full_list - marker_list

    line_list = []
    for marker in missing_list:
        line_list.extend(missing_marker_mapper[marker])

    new_lines = []
    for i in range(len(lines)):
        line = lines[i]
        if i in line_list:
            line = line.replace(line,f'<div style="background-color: lightgreen;">{line}</div>')
        new_lines.append(line)
    text = "<TEXT-SPLITTER>".join(new_lines)
    return text

def get_annotation_mvp_v2(match_examples, level):
    missing_marker_mapper = {"HOSPITAL":[11], "MALARIA":[12,13], "FARMING":[16], "SCHOOL":[17]}
    markers = ["GENERAL", "GENERAL", "HOSPITAL", "MALARIA", "FARMING", "SCHOOL", "HOSPITAL", "GENERAL"]
    with open(os.path.join(settings.BASE_DIR, "templates/mvp_essay.html"), "r") as f:
        lines = f.readlines()

    marker_list = []
    for i in range(len(match_examples)):
        if len(match_examples[i]) > 0:
            marker = markers[i]
            marker_list.append(marker)

    full_list = set(["HOSPITAL", "MALARIA", "FARMING", "SCHOOL"])
    marker_list = set(marker_list)
    missing_list = full_list - marker_list

    line_list = []
    for marker in missing_list:
        line_list.extend(missing_marker_mapper[marker])

    new_lines = []
    new_line_num = []
    for i in range(len(lines)):
        line = lines[i]
        if i in line_list and level==1:
            line = line.replace(line,f'<div style="background-color: lightgreen; border-radius: 6px;">{line}</div>')
            new_line_num.append(i)
        new_lines.append(line)
    text = "<TEXT-SPLITTER>".join(new_lines)
    return text, new_line_num

def get_annotation_space_v2(match_examples, level):
    missing_marker_mapper = {"HELP_PEOPLE":[4,5], "HELP_ENVIRONMENT":[6,7], "TANGIBLE_BENEFITS":[9,10,11,12], "SPIRIT_OF_EXPLORATION":[14,15]}
    markers = ["HELP_PEOPLE", "HELP_PEOPLE", "HELP_ENVIRONMENT", "HELP_ENVIRONMENT", "TANGIBLE_BENEFITS", "TANGIBLE_BENEFITS", "SPIRIT_OF_EXPLORATION", "SPIRIT_OF_EXPLORATION"]
    with open(os.path.join(settings.BASE_DIR, "templates/space_essay.html"), "r") as f:
        lines = f.readlines()

    marker_list = []
    for i in range(len(match_examples)):
        if len(match_examples[i]) > 0:
            marker = markers[i]
            marker_list.append(marker)

    full_list = set(["HELP_PEOPLE", "HELP_ENVIRONMENT", "TANGIBLE_BENEFITS", "SPIRIT_OF_EXPLORATION"])
    marker_list = set(marker_list)
    missing_list = full_list - marker_list

    line_list = []
    for marker in missing_list:
        line_list.extend(missing_marker_mapper[marker])

    new_lines = []
    new_line_num = []
    for i in range(len(lines)):
        line = lines[i]
        if i in line_list and level==1:
            line = line.replace(line,f'<div style="background-color: lightgreen; border-radius: 6px;">{line}</div>')
            new_line_num.append(i)
        new_lines.append(line)
    text = "<TEXT-SPLITTER>".join(new_lines)
    return text, new_line_num


def process_essays(request):

    if request.user.is_authenticated and request.user.is_staff:

        submission = Essay.objects.values('user_name', 'essay_revision', 'prompt_name').distinct()
        latest_records = []
        for distinct_value in submission:
            # Retrieve the latest record for each distinct value in the 'field_name' field
            latest_record = Essay.objects.filter(user_name=distinct_value['user_name'],
                                                 essay_revision=distinct_value['essay_revision'],
                                                 prompt_name=distinct_value['prompt_name'],).order_by(
                'submitted_time').last()
            if latest_record and latest_record.processed==False:
                latest_records.append(latest_record)

        for record in latest_records:
            user_name = record.user_name
            essay_revision = record.essay_revision
            prompt_name = record.prompt_name
            essay_content = record.essay_content

            score_essay = ScoreEssay(essay_content, topic=prompt_name, threshold=0.9)
            level, npe, spc, match_examples, input_topic_mapper = score_essay.get_feedback_level()
            if essay_revision > 1:
                if level[0] in [3,4] or level[1] in [3,4]:
                    level = [4, 5]
            feedback0 = get_feedback(prompt_name, level[0], npe, spc)
            feedback1 = get_feedback(prompt_name, level[1], npe, spc)
            feedback = feedback0 + feedback1
            if prompt_name == "MVP":
                annotating_text, annotating_line_number = get_annotation_mvp_v2(match_examples, level[0])
            elif prompt_name == "SPACE":
                annotating_text, annotating_line_number = get_annotation_space_v2(match_examples, level[0])
            else:
                raise "Error"
            now = timezone.now()

            data = Process(
                user_name=user_name,
                prompt_name=prompt_name,
                essay_content=essay_content,
                essay_revision=essay_revision,
                auto_feedback=feedback,
                teacher_feedback='',
                npe_score = npe,
                spc_score = spc,
                annotating_text=annotating_text,
                processed_time=now,
            )
            data.save()

            record.processed = True
            record.processed_time = now
            record.save()
    return render(request, 'index.html')


def get_revision_content(revision_list, user_name, prompt_name, essay_revision):
    latest_record = Essay.objects.filter(user_name=user_name,
                                         essay_revision=essay_revision,
                                         prompt_name=prompt_name).order_by('submitted_time').last()
    if not latest_record:
        return ""
    else:
        return latest_record.essay_content
def process_essays_v2(request):

    if request.user.is_authenticated and request.user.is_staff:

        submission = Essay.objects.values('user_name', 'essay_revision', 'prompt_name').distinct()
        latest_records = []
        for distinct_value in submission:
            # Retrieve the latest record for each distinct value in the 'field_name' field
            latest_record = Essay.objects.filter(user_name=distinct_value['user_name'],
                                                 essay_revision=distinct_value['essay_revision'],
                                                 prompt_name=distinct_value['prompt_name'],).order_by('submitted_time').last()
            if latest_record and latest_record.processed==False:
                latest_records.append(latest_record)


        for record in latest_records:
            user_name = record.user_name
            essay_revision = record.essay_revision
            prompt_name = record.prompt_name
            essay_content = record.essay_content

            if essay_revision == 1 or essay_revision == 3:
                score_essay = ScoreEssay(essay_content, topic=prompt_name, threshold=0.9)
                old_evidence_level, old_npe, old_spc, old_match_examples, old_input_topic_mapper = score_essay.get_feedback_level_v2()
                feedback = get_evidence_feedback(prompt_name, old_evidence_level)
                npe = old_npe
                spc = old_spc
                npe_keyword = str(old_input_topic_mapper)
                spc_keyword = str(old_match_examples)
                fb_level = old_evidence_level

                if prompt_name == "MVP":
                    annotating_text, annotating_line_number = get_annotation_mvp_v2(old_match_examples, old_evidence_level)
                elif prompt_name == "SPACE":
                    annotating_text, annotating_line_number = get_annotation_space_v2(old_match_examples, old_evidence_level)
                else:
                    raise "Error"

            elif essay_revision == 2:
                old_draft = get_revision_content(latest_records, user_name, prompt_name, essay_revision-1)
                new_draft = essay_content

                old_score_essay = ScoreEssay(old_draft, topic=prompt_name, threshold=0.9)
                old_evidence_level, old_npe, old_spc, old_match_examples, old_input_topic_mapper = old_score_essay.get_feedback_level_v2()

                new_score_essay = ScoreEssay(new_draft, topic=prompt_name, threshold=0.9)
                new_evidence_level, new_npe, new_spc, new_match_examples, new_input_topic_mapper = new_score_essay.get_feedback_level_v2()

                score_revision = ScoreRevision(old_draft, new_draft, topic=prompt_name)
                score_revision.align_document()
                score_revision.get_argument_context()
                score_revision.predict_successfulness()
                score_revision.predict_evidence_reasoning()
                score_revision.predict_topic_relevance()
                revision_level = score_revision.get_predict_level(old_evidence_level, old_npe, new_npe, old_spc, new_spc)

                feedback = get_revision_feedback(prompt_name, revision_level, old_evidence_level)
                npe = new_npe
                spc = new_spc
                npe_keyword = str(new_input_topic_mapper)
                spc_keyword = str(new_match_examples)
                fb_level = revision_level

                if prompt_name == "MVP":
                    annotating_text, annotating_line_number = get_annotation_mvp_v2(old_match_examples, old_evidence_level)
                elif prompt_name == "SPACE":
                    annotating_text, annotating_line_number = get_annotation_space_v2(old_match_examples, old_evidence_level)
                else:
                    raise "Error"

                revision_records = score_revision.master_df
                revision_records["old_npe_score"] = old_npe
                revision_records["new_npe_score"] = new_npe
                revision_records["old_spc_score"] = old_spc
                revision_records["new_spc_score"] = new_spc
                revision_records["old_npe_keyword"] = str(old_input_topic_mapper)
                revision_records["new_npe_keyword"] = str(new_input_topic_mapper)
                revision_records["old_spc_keyword"] = str(old_match_examples)
                revision_records["new_spc_keyword"] = str(new_match_examples)
                revision_records["auto_feedback"] = feedback
                revision_records["evidence_feedback_level"] = old_evidence_level
                revision_records["revision_feedback_level"] = revision_level
                revision_records["annotating_text"] = annotating_text
                revision_records["annotating_line_number"] = str(annotating_line_number)

                for k in range(len(revision_records)):
                    old_sentence_id = revision_records["old_sentence_id"][k]
                    old_sentence_aligned_id = revision_records["old_sentence_aligned_id"][k]
                    old_sentence = revision_records["old_sentence"][k]
                    new_sentence = revision_records["new_sentence"][k]
                    new_sentence_id = revision_records["new_sentence_id"][k]
                    new_sentence_aligned_id = revision_records["new_sentence_aligned_id"][k]
                    coarse_label = revision_records["coarse_label"][k]
                    old_argument_context = revision_records["old_argument_context"][k]
                    new_argument_context = revision_records["new_argument_context"][k]
                    used_context = revision_records["used_context"][k]
                    used_sentence = revision_records["used_revision"][k]
                    successfulness = revision_records["successfulness"][k]
                    fine_label = revision_records["fine_label"][k]
                    mentioned_keyword = revision_records["mentioned_topic"][k]
                    old_npe = revision_records["old_npe_score"][k]
                    new_npe = revision_records["new_npe_score"][k]
                    old_spc = revision_records["old_spc_score"][k]
                    new_spc = revision_records["new_spc_score"][k]
                    old_npe_keyword = revision_records["old_npe_keyword"][k]
                    new_npe_keyword = revision_records["new_npe_keyword"][k]
                    old_spc_keyword = revision_records["old_spc_keyword"][k]
                    new_spc_keyword = revision_records["new_spc_keyword"][k]
                    auto_feedback = revision_records["auto_feedback"][k]
                    evidence_feedback_level = revision_records["evidence_feedback_level"][k]
                    revision_feedback_level = revision_records["revision_feedback_level"][k]
                    annotating_text = revision_records["annotating_text"][k]
                    annotating_line_number = revision_records["annotating_line_number"][k]
                    now = timezone.now()


                    data = ProcessRevision(
                        user_name=user_name,
                        prompt_name=prompt_name,
                        essay_revision=essay_revision,
                        old_essay=old_draft,
                        new_essay=new_draft,
                        old_sentence_id=old_sentence_id,
                        old_sentence_aligned_id=old_sentence_aligned_id,
                        old_sentence=old_sentence,
                        new_sentence=new_sentence,
                        new_sentence_id=new_sentence_id,
                        new_sentence_aligned_id=new_sentence_aligned_id,
                        coarse_label=coarse_label,
                        old_argument_context=old_argument_context,
                        new_argument_context=new_argument_context,
                        used_context=used_context,
                        used_sentence=used_sentence,
                        successfulness=successfulness,
                        fine_label=fine_label,
                        mentioned_topic=mentioned_keyword,
                        old_npe_score=old_npe,
                        new_npe_score=new_npe,
                        old_spc_score=old_spc,
                        new_spc_score=new_spc,
                        old_npe_keyword=old_npe_keyword,
                        new_npe_keyword=new_npe_keyword,
                        old_spc_keyword=old_spc_keyword,
                        new_spc_keyword=new_spc_keyword,
                        auto_feedback=auto_feedback,
                        evidence_feedback_level=evidence_feedback_level,
                        revision_feedback_level=revision_feedback_level,
                        annotating_text=annotating_text,
                        annotating_line_number=annotating_line_number,
                        processed_time=now,
                    )

                    data.save()


            else:
                continue

            now = timezone.now()

            data = Process(
                user_name=user_name,
                prompt_name=prompt_name,
                essay_content=essay_content,
                essay_revision=essay_revision,
                auto_feedback=feedback,
                feedback_level=fb_level,
                teacher_feedback='',
                npe_score=npe,
                spc_score=spc,
                npe_keyword=npe_keyword,
                spc_keyword=spc_keyword,
                annotating_text=annotating_text,
                annotating_line_number=annotating_line_number,
                processed_time=now,
            )
            data.save()

            record.processed = True
            record.processed_time = now
            record.save()
    return render(request, 'index.html')

def show_mvp(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')
    return render(request, 'mvp_article.html')
def show_space(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')
    return render(request, 'space_article.html')

def submit_mvp(request):
    # prompt_name = request.POST['prompt_name']
    if not request.user.is_authenticated:
        return render(request, 'index.html')

    revision = RevisionForm(request.POST)
    form = EssayForm(request.POST)

    if revision.is_valid() and request.method == 'POST':
        revision_id = revision.cleaned_data["essay_revision"]
        revision_id = int(revision_id) - 1
    else:
        revision_id = 0
    if revision_id >=1:
        submited_essay =  get_essay_revision("MVP", revision_id)
        score_essay = ScoreEssay(submited_essay, topic="MVP", threshold=0.9)
        level, npe, spc, match_examples, input_topic_mapper = score_essay.get_feedback_level()
        write_color_essay_html_v2(match_examples)
        write_color_input_html(submited_essay, input_topic_mapper)
    else:
        submited_essay = None
        level, npe, spc = None, None, None

    prompt_name = "MVP"
    # revision["submited_essay"] = submited_essay
    init_prompt = get_prompt(prompt_name="MVP")
    if request.method == 'POST':
        essay = request.POST['essay']
        if form.is_valid() and len(essay)>=1:
            user_name = request.user.username


            # essay_revision = request.POST['essay_revision']
            # now = datetime.datetime.now()
            now = timezone.now()
            data = Essay(
                user_name=user_name,
                prompt_name=prompt_name,
                essay_content=essay,
                essay_revision=revision_id+1,
                submitted=True,
                processed=False,
                submitted_time=now,
                processed_time=now,
                created_time=now,
                modified_time=now
            )
            data.save()
            score_essay = ScoreEssay(essay, topic="MVP", threshold=0.9)
            level, npe, spc, match_examples, input_topic_mapper = score_essay.get_feedback_level()
            write_color_essay_html_v2(match_examples)

            feedback0 = get_feedback(prompt_name, level[0], npe, spc)
            feedback1 = get_feedback(prompt_name, level[1], npe, spc)
            form = EssayForm()

            submited_essay = get_essay_revision("MVP", revision_id+1)
            write_color_input_html(submited_essay, input_topic_mapper)
            return render(request, 'mvp.html', {'form': form, "init_prompt": init_prompt, "feedback_list":[feedback0, feedback1], "revision": revision, "submited_essay":submited_essay, "successful": True, 'user_name': user_name})

        else:
            if revision_id >=1:
                feedback0 = get_feedback(prompt_name, level[0], npe, spc)
                feedback1 = get_feedback(prompt_name, level[1], npe, spc)
                return render(request, 'mvp.html',
                              {'form': form, "init_prompt": init_prompt, "feedback_list": [feedback0, feedback1],
                               "revision": revision, "submited_essay": submited_essay, "successful": False, 'user_name':''})
            else:
                return render(request, 'mvp.html', {'form': form, "init_prompt": init_prompt, "revision": revision,
                                                       "submited_essay": submited_essay, "successful": False, 'user_name':''})

    # else:
    #     form = EssayForm()
    # highlight match tokens



    if revision_id >=1:
        feedback0 = get_feedback(prompt_name, level[0], npe, spc)
        feedback1 = get_feedback(prompt_name, level[1], npe, spc)
        return render(request, 'mvp.html', {'form': form, "init_prompt": init_prompt, "feedback_list":[feedback0, feedback1], "revision": revision, "submited_essay":submited_essay, "successful": False, 'user_name':''})
    else:
        return render(request, 'mvp.html', {'form': form, "init_prompt": init_prompt, "revision": revision, "submited_essay":submited_essay, "successful": False, 'user_name':''})


def submit_mvp_v2(request):

    if not request.user.is_authenticated:
        return render(request, 'index.html')

    # prompt_name = request.POST['prompt_name']
    essay_revision = 0
    level = 0
    prompt_name = "MVP"
    user_name = request.user.username

    latest_entry = Essay.objects.filter(user_name=user_name, prompt_name=prompt_name, processed=True).order_by('submitted_time').last()
    if not latest_entry:
        essay_revision = 0
        feedback_list = []
        teacher_feedback = None
    else:
        essay_revision = latest_entry.essay_revision
        processed_entry = Process.objects.filter(user_name=user_name, prompt_name=prompt_name, essay_revision=essay_revision).order_by('processed_time').last()
        auto_feedback = processed_entry.auto_feedback

        parsed_html = parse_feedback(auto_feedback)

        feedback_list = [{"title": parsed_html["title"][0], "content":parsed_html["content"][0]},
                         {"title": parsed_html["title"][1], "content":parsed_html["content"][1]}]

        teacher_feedback = processed_entry.teacher_feedback

    if essay_revision >=1:
        previous_revision = get_essay_revision("MVP", essay_revision)
    else:
        previous_revision = None

    form = EssayForm(request.POST)

    # if revision.is_valid() and request.method == 'POST':
    #     essay_revision = revision.cleaned_data["essay_revision"]
    #     essay_revision = int(essay_revision) - 1
    # else:
    #     essay_revision = 0
    if essay_revision >=1:
        submited_essay =  get_essay_revision("MVP", essay_revision)
        score_essay = ScoreEssay(submited_essay, topic="MVP", threshold=0.9)
        level, npe, spc, match_examples, input_topic_mapper = score_essay.get_feedback_level()
        level = level[0]
        write_color_essay_html_v2(match_examples)
        write_color_input_html(submited_essay, input_topic_mapper)
    # else:


    # revision["submited_essay"] = submited_essay
    init_prompt = get_prompt(prompt_name="MVP")
    if request.method == 'POST':
        essay = request.POST['essay']
        if form.is_valid() and len(essay)>=1:
            now = timezone.now()
            data = Essay(
                user_name=user_name,
                prompt_name=prompt_name,
                essay_content=essay,
                essay_revision=essay_revision+1,
                submitted=True,
                processed=False,
                submitted_time=now,
                processed_time=now,
                created_time=now,
                modified_time=now
            )
            data.save()

            return render(request, 'mvp.html', {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list, "teacher_feedback":teacher_feedback,
                                                "revision": essay_revision+1, "previous_revision":previous_revision, "successful": True, 'user_name': user_name, "level":level})

        else:
            return render(request, 'mvp.html',
                          {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list,
                           "teacher_feedback": teacher_feedback,
                           "revision": essay_revision + 1, "previous_revision": previous_revision, "successful": False,
                           'user_name': user_name, "level":level})

    # else:
    #     form = EssayForm()
    # highlight match tokens

    else:
        return render(request, 'mvp.html', {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list,
                                            "teacher_feedback": teacher_feedback,
                                            "revision": essay_revision + 1, "previous_revision": previous_revision,
                                            "successful": False, 'user_name': user_name, "level":level})


def submit_mvp_v3(request):

    if not request.user.is_authenticated:
        return render(request, 'index.html')
    # prompt_name = request.POST['prompt_name']
    essay_revision = 0
    level = 0
    prompt_name = "MVP"
    user_name = request.user.username
    annotating_text = ""

    latest_entry = Essay.objects.filter(user_name=user_name, prompt_name=prompt_name, processed=True).order_by('submitted_time').last()
    if not latest_entry:
        essay_revision = 0
        feedback_list = []
        teacher_feedback = None
    else:
        essay_revision = latest_entry.essay_revision
        processed_entry = Process.objects.filter(user_name=user_name, prompt_name=prompt_name, essay_revision=essay_revision).order_by('processed_time').last()
        auto_feedback = processed_entry.auto_feedback

        parsed_html = parse_feedback(auto_feedback)

        feedback_list = [{"title": parsed_html["title"][0], "content":parsed_html["content"][0]},
                         {"title": parsed_html["title"][1], "content":parsed_html["content"][1]}]

        teacher_feedback = processed_entry.teacher_feedback
        annotating_text = processed_entry.annotating_text

    if essay_revision >=1:
        previous_revision = get_essay_revision_v2(user_name, "MVP", essay_revision)
    else:
        previous_revision = None

    form = EssayForm(request.POST)

    # if revision.is_valid() and request.method == 'POST':
    #     essay_revision = revision.cleaned_data["essay_revision"]
    #     essay_revision = int(essay_revision) - 1
    # else:
    #     essay_revision = 0
    # if essay_revision >=1:
    #     submited_essay =  get_essay_revision_v2(user_name,"MVP", essay_revision)
    #     score_essay = ScoreEssay(submited_essay, topic="MVP", threshold=0.9)
    #     level, npe, spc, match_examples, input_topic_mapper = score_essay.get_feedback_level()
    #     level = level[0]
    #     write_color_essay_html_v2(match_examples)
    #     write_color_input_html(submited_essay, input_topic_mapper)
    # else:


    # revision["submited_essay"] = submited_essay
    init_prompt = get_prompt(prompt_name="MVP")
    if request.method == 'POST':
        essay = request.POST['essay']
        if form.is_valid() and len(essay)>=1:
            now = timezone.now()
            data = Essay(
                user_name=user_name,
                prompt_name=prompt_name,
                essay_content=essay,
                essay_revision=essay_revision+1,
                submitted=True,
                processed=False,
                submitted_time=now,
                processed_time=now,
                created_time=now,
                modified_time=now
            )
            data.save()

            return render(request, 'mvp.html', {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list, "teacher_feedback":teacher_feedback,
                                                "revision": essay_revision+1, "previous_revision":previous_revision, "successful": True, 'user_name': user_name, "level":level, "annotating_text":annotating_text})

        else:
            return render(request, 'mvp.html',
                          {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list,
                           "teacher_feedback": teacher_feedback,
                           "revision": essay_revision + 1, "previous_revision": previous_revision, "successful": False,
                           'user_name': user_name, "level":level, "annotating_text":annotating_text})

    # else:
    #     form = EssayForm()
    # highlight match tokens

    else:
        return render(request, 'mvp.html', {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list,
                                            "teacher_feedback": teacher_feedback,
                                            "revision": essay_revision + 1, "previous_revision": previous_revision,
                                            "successful": False, 'user_name': user_name, "level":level, "annotating_text":annotating_text})

def submit_mvp_v4(request):

    if not request.user.is_authenticated:
        return render(request, 'index.html')
    # prompt_name = request.POST['prompt_name']
    essay_revision = 0
    level = 0
    prompt_name = "MVP"
    user_name = request.user.username
    annotating_text = ""

    latest_entry = Essay.objects.filter(user_name=user_name, prompt_name=prompt_name, processed=True).order_by('submitted_time').last()
    if not latest_entry:
        essay_revision = 0
        feedback_list = []
        teacher_feedback = None
    else:
        essay_revision = latest_entry.essay_revision
        processed_entry = Process.objects.filter(user_name=user_name, prompt_name=prompt_name, essay_revision=essay_revision).order_by('processed_time').last()
        auto_feedback = processed_entry.auto_feedback

        # parsed_html = parse_feedback(auto_feedback)
        #
        # feedback_list = [{"title": parsed_html["title"][0], "content":parsed_html["content"][0]},
        #                  {"title": parsed_html["title"][1], "content":parsed_html["content"][1]}]

        feedback_list = auto_feedback
        teacher_feedback = processed_entry.teacher_feedback
        annotating_text = processed_entry.annotating_text

    if essay_revision >=1:
        previous_revision = get_essay_revision_v2(user_name, "MVP", essay_revision)
    else:
        previous_revision = None

    form = EssayForm(request.POST)

    # if revision.is_valid() and request.method == 'POST':
    #     essay_revision = revision.cleaned_data["essay_revision"]
    #     essay_revision = int(essay_revision) - 1
    # else:
    #     essay_revision = 0
    # if essay_revision >=1:
    #     submited_essay =  get_essay_revision_v2(user_name,"MVP", essay_revision)
    #     score_essay = ScoreEssay(submited_essay, topic="MVP", threshold=0.9)
    #     level, npe, spc, match_examples, input_topic_mapper = score_essay.get_feedback_level()
    #     level = level[0]
    #     write_color_essay_html_v2(match_examples)
    #     write_color_input_html(submited_essay, input_topic_mapper)
    # else:


    # revision["submited_essay"] = submited_essay
    init_prompt = get_prompt(prompt_name="MVP")
    if request.method == 'POST':
        essay = request.POST['essay']
        if form.is_valid() and len(essay)>=1:
            now = timezone.now()
            data = Essay(
                user_name=user_name,
                prompt_name=prompt_name,
                essay_content=essay,
                essay_revision=essay_revision+1,
                submitted=True,
                processed=False,
                submitted_time=now,
                processed_time=now,
                created_time=now,
                modified_time=now
            )
            data.save()

            return render(request, 'mvp.html', {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list, "teacher_feedback":teacher_feedback,
                                                "revision": essay_revision+1, "previous_revision":previous_revision, "successful": True, 'user_name': user_name, "level":level, "annotating_text":annotating_text})

        else:
            return render(request, 'mvp.html',
                          {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list,
                           "teacher_feedback": teacher_feedback,
                           "revision": essay_revision + 1, "previous_revision": previous_revision, "successful": False,
                           'user_name': user_name, "level":level, "annotating_text":annotating_text})

    # else:
    #     form = EssayForm()
    # highlight match tokens

    else:
        return render(request, 'mvp.html', {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list,
                                            "teacher_feedback": teacher_feedback,
                                            "revision": essay_revision + 1, "previous_revision": previous_revision,
                                            "successful": False, 'user_name': user_name, "level":level, "annotating_text":annotating_text})

def submit_space(request):

    if not request.user.is_authenticated:
        return render(request, 'index.html')
    # prompt_name = request.POST['prompt_name']
    revision = RevisionForm(request.POST)
    form = EssayForm(request.POST)

    if revision.is_valid() and request.method == 'POST':
        revision_id = revision.cleaned_data["essay_revision"]
        revision_id = int(revision_id) - 1
    else:
        revision_id = 0
    if revision_id >=1:
        submited_essay = get_essay_revision("SPACE", revision_id)
        score_essay = ScoreEssay(submited_essay, topic="SPACE", threshold=0.9)
        level, npe, spc, match_examples, input_topic_mapper = score_essay.get_feedback_level()
        write_color_essay_html_space(match_examples)
        write_color_input_html_space(submited_essay, input_topic_mapper)
    else:
        submited_essay = None
        level, npe, spc = None, None, None

    prompt_name = "SPACE"
    # revision["submited_essay"] = submited_essay
    init_prompt = get_prompt(prompt_name="SPACE")
    if request.method == 'POST':
        essay = request.POST['essay']
        if form.is_valid() and len(essay)>=1:
            user_name = request.user.username


            # essay_revision = request.POST['essay_revision']
            # now = datetime.datetime.now()
            now = timezone.now()
            data = Essay(
                user_name=user_name,
                prompt_name=prompt_name,
                essay_content=essay,
                essay_revision=revision_id+1,
                submitted=True,
                processed=False,
                submitted_time=now,
                processed_time=now,
                created_time=now,
                modified_time=now
            )
            data.save()
            score_essay = ScoreEssay(essay, topic="SPACE", threshold=0.9)
            level, npe, spc, match_examples, input_topic_mapper = score_essay.get_feedback_level()
            write_color_essay_html_space(match_examples)

            feedback0 = get_feedback(prompt_name, level[0], npe, spc)
            feedback1 = get_feedback(prompt_name, level[1], npe, spc)
            form = EssayForm()

            submited_essay = get_essay_revision("SPACE", revision_id+1)
            write_color_input_html_space(submited_essay, input_topic_mapper)
            return render(request, 'space.html', {'form': form, "init_prompt": init_prompt, "feedback_list":[feedback0, feedback1], "revision": revision, "submited_essay":submited_essay, "successful": True, 'user_name':user_name})

        else:
            if revision_id >=1:
                feedback0 = get_feedback(prompt_name, level[0], npe, spc)
                feedback1 = get_feedback(prompt_name, level[1], npe, spc)
                return render(request, 'space.html',
                              {'form': form, "init_prompt": init_prompt, "feedback_list": [feedback0, feedback1],
                               "revision": revision, "submited_essay": submited_essay, "successful": False, 'user_name':''})
            else:
                return render(request, 'space.html', {'form': form, "init_prompt": init_prompt, "revision": revision,
                                                       "submited_essay": submited_essay, "successful": False, 'user_name':''})

    # else:
    #     form = EssayForm()
    # highlight match tokens



    if revision_id >=1:
        feedback0 = get_feedback(prompt_name, level[0], npe, spc)
        feedback1 = get_feedback(prompt_name, level[1], npe, spc)
        return render(request, 'space.html', {'form': form, "init_prompt": init_prompt, "feedback_list":[feedback0, feedback1], "revision": revision, "submited_essay":submited_essay, "successful": False, 'user_name':''})
    else:
        return render(request, 'space.html', {'form': form, "init_prompt": init_prompt, "revision": revision, "submited_essay":submited_essay, "successful": False, 'user_name':''})

def submit_space_v2(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')
    # prompt_name = request.POST['prompt_name']
    essay_revision = 0
    level = 0
    prompt_name = "SPACE"
    user_name = request.user.username

    latest_entry = Essay.objects.filter(user_name=user_name, prompt_name=prompt_name, processed=True).order_by('submitted_time').last()
    if not latest_entry:
        essay_revision = 0
        feedback_list = []
        teacher_feedback = None
    else:
        essay_revision = latest_entry.essay_revision
        processed_entry = Process.objects.filter(user_name=user_name, prompt_name=prompt_name, essay_revision=essay_revision).order_by('processed_time').last()
        auto_feedback = processed_entry.auto_feedback

        parsed_html = parse_feedback(auto_feedback)

        feedback_list = [{"title": parsed_html["title"][0], "content":parsed_html["content"][0]},
                         {"title": parsed_html["title"][1], "content":parsed_html["content"][1]}]

        teacher_feedback = processed_entry.teacher_feedback

    if essay_revision >=1:
        previous_revision = get_essay_revision("SPACE", essay_revision)
    else:
        previous_revision = None

    form = EssayForm(request.POST)

    # if revision.is_valid() and request.method == 'POST':
    #     essay_revision = revision.cleaned_data["essay_revision"]
    #     essay_revision = int(essay_revision) - 1
    # else:
    #     essay_revision = 0
    if essay_revision >=1:
        submited_essay =  get_essay_revision("SPACE", essay_revision)
        score_essay = ScoreEssay(submited_essay, topic="SPACE", threshold=0.9)
        level, npe, spc, match_examples, input_topic_mapper = score_essay.get_feedback_level()
        level = level[0]
        write_color_essay_html_space(match_examples)
        write_color_input_html_space(submited_essay, input_topic_mapper)
    # else:


    # revision["submited_essay"] = submited_essay
    init_prompt = get_prompt(prompt_name="SPACE")
    if request.method == 'POST':
        essay = request.POST['essay']
        if form.is_valid() and len(essay)>=1:
            now = timezone.now()
            data = Essay(
                user_name=user_name,
                prompt_name=prompt_name,
                essay_content=essay,
                essay_revision=essay_revision+1,
                submitted=True,
                processed=False,
                submitted_time=now,
                processed_time=now,
                created_time=now,
                modified_time=now
            )
            data.save()

            return render(request, 'space.html', {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list, "teacher_feedback":teacher_feedback,
                                                "revision": essay_revision+1, "previous_revision":previous_revision, "successful": True, 'user_name': user_name, "level":level})

        else:
            return render(request, 'space.html',
                          {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list,
                           "teacher_feedback": teacher_feedback,
                           "revision": essay_revision + 1, "previous_revision": previous_revision, "successful": False,
                           'user_name': user_name, "level":level})

    # else:
    #     form = EssayForm()
    # highlight match tokens

    else:
        return render(request, 'space.html', {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list,
                                            "teacher_feedback": teacher_feedback,
                                            "revision": essay_revision + 1, "previous_revision": previous_revision,
                                            "successful": False, 'user_name': user_name, "level":level})

def submit_space_v3(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')
    # prompt_name = request.POST['prompt_name']
    essay_revision = 0
    level = 0
    prompt_name = "SPACE"
    user_name = request.user.username
    annotating_text = ""

    latest_entry = Essay.objects.filter(user_name=user_name, prompt_name=prompt_name, processed=True).order_by('submitted_time').last()
    if not latest_entry:
        essay_revision = 0
        feedback_list = []
        teacher_feedback = None
    else:
        essay_revision = latest_entry.essay_revision
        processed_entry = Process.objects.filter(user_name=user_name, prompt_name=prompt_name, essay_revision=essay_revision).order_by('processed_time').last()
        auto_feedback = processed_entry.auto_feedback

        parsed_html = parse_feedback(auto_feedback)

        feedback_list = [{"title": parsed_html["title"][0], "content":parsed_html["content"][0]},
                         {"title": parsed_html["title"][1], "content":parsed_html["content"][1]}]

        teacher_feedback = processed_entry.teacher_feedback
        annotating_text = processed_entry.annotating_text

    if essay_revision >=1:
        previous_revision = get_essay_revision_v2(user_name, "SPACE", essay_revision)
    else:
        previous_revision = None

    form = EssayForm(request.POST)

    # if revision.is_valid() and request.method == 'POST':
    #     essay_revision = revision.cleaned_data["essay_revision"]
    #     essay_revision = int(essay_revision) - 1
    # else:
    #     essay_revision = 0
    # if essay_revision >=1:
    #     submited_essay =  get_essay_revision_v2(user_name,"MVP", essay_revision)
    #     score_essay = ScoreEssay(submited_essay, topic="MVP", threshold=0.9)
    #     level, npe, spc, match_examples, input_topic_mapper = score_essay.get_feedback_level()
    #     level = level[0]
    #     write_color_essay_html_v2(match_examples)
    #     write_color_input_html(submited_essay, input_topic_mapper)
    # else:


    # revision["submited_essay"] = submited_essay
    init_prompt = get_prompt(prompt_name="SPACE")
    if request.method == 'POST':
        essay = request.POST['essay']
        if form.is_valid() and len(essay)>=1:
            now = timezone.now()
            data = Essay(
                user_name=user_name,
                prompt_name=prompt_name,
                essay_content=essay,
                essay_revision=essay_revision+1,
                submitted=True,
                processed=False,
                submitted_time=now,
                processed_time=now,
                created_time=now,
                modified_time=now
            )
            data.save()

            return render(request, 'space.html', {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list, "teacher_feedback":teacher_feedback,
                                                "revision": essay_revision+1, "previous_revision":previous_revision, "successful": True, 'user_name': user_name, "level":level, "annotating_text":annotating_text})

        else:
            return render(request, 'space.html',
                          {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list,
                           "teacher_feedback": teacher_feedback,
                           "revision": essay_revision + 1, "previous_revision": previous_revision, "successful": False,
                           'user_name': user_name, "level":level, "annotating_text":annotating_text})

    # else:
    #     form = EssayForm()
    # highlight match tokens

    else:
        return render(request, 'space.html', {'form': form, "init_prompt": init_prompt, "feedback_list": feedback_list,
                                            "teacher_feedback": teacher_feedback,
                                            "revision": essay_revision + 1, "previous_revision": previous_revision,
                                            "successful": False, 'user_name': user_name, "level":level, "annotating_text":annotating_text})

def write_color_essay_html(match_examples):
    with open(os.path.join(settings.BASE_DIR, "templates/mvp_essay.html"), "r") as f:
        lines = f.readlines()

    with open(os.path.join(settings.BASE_DIR, "statics/data/MVP/examples_map.json"), "r") as f:
        mapper = json.load(f)

    new_lines = []
    colors = ["#00FEFE", "#00FF00", "yellow", "#FF00FF",  "#00CDFF", "#FF00FF"]
    color_marker_mapper = [0, 0, 1, 2, 3, 4 ,2, 0]
    markers = ["GENERAL", "GENERAL", "HOSPITAL", "MALARIA", "FARMING", "SCHOOL", "HOSPITAL", "GENERAL"]

    for line in lines:
        for i in range(len(match_examples)):
            color = colors[color_marker_mapper[i]]
            marker = markers[i]
            phrase_list = match_examples[i]
            phrase_set = set()
            for j in range(len(phrase_list)):
                phrase = " ".join(phrase_list[j])
                phrase_set.add(phrase)
            for phrase in phrase_set:
                if phrase in mapper:
                    full_phrase = mapper[phrase]
                    line = line.replace(full_phrase, f'<span style="background-color:{color}">{full_phrase}<sub>[{marker}]</sub></span>')
        new_lines.append(line)

    with open(os.path.join(settings.BASE_DIR, "templates/mvp_essay_color.html"), "w") as f:
        f.writelines(new_lines)

def write_color_essay_html_v2(match_examples):
    with open(os.path.join(settings.BASE_DIR, "templates/mvp_essay.html"), "r") as f:
        lines = f.readlines()

    with open(os.path.join(settings.BASE_DIR, "statics/data/MVP/examples_map.json"), "r") as f:
        mapper = json.load(f)

    new_lines = []
    colors = ["#00FEFE", "#00FF00", "yellow", "#FF00FF",  "#00CDFF", "#FF00FF"]
    color_marker_mapper = [0, 0, 1, 2, 3, 4 ,2, 0]
    markers = ["GENERAL", "GENERAL", "HOSPITAL", "MALARIA", "FARMING", "SCHOOL", "HOSPITAL", "GENERAL"]
    missing_marker_mapper = {"HOSPITAL":[11], "MALARIA":[12,13], "FARMING":[16], "SCHOOL":[17]}

    marker_list = []
    for i in range(len(match_examples)):
        if len(match_examples[i]) > 0:
            marker = markers[i]
            marker_list.append(marker)

    full_list = set(["HOSPITAL", "MALARIA", "FARMING", "SCHOOL"])
    marker_list = set(marker_list)
    missing_list = full_list - marker_list

    line_list = []
    for marker in missing_list:
        line_list.extend(missing_marker_mapper[marker])


    for i in range(len(lines)):
        line = lines[i]
        if i in line_list:
            line = line.replace(line,f'<div style="background-color: lightgreen;">{line}</div>')
        new_lines.append(line)

    with open(os.path.join(settings.BASE_DIR, "templates/mvp_essay_color.html"), "w") as f:
        f.writelines(new_lines)


def write_color_essay_html_space(match_examples):
    with open(os.path.join(settings.BASE_DIR, "templates/space_essay.html"), "r") as f:
        lines = f.readlines()

    with open(os.path.join(settings.BASE_DIR, "statics/data/SPACE/examples_map.json"), "r") as f:
        mapper = json.load(f)

    new_lines = []
    colors = ["#00FF00", "yellow", "#FF00FF",  "#00CDFF",]
    color_marker_mapper = [0, 0, 1, 1, 2, 2, 3, 3]
    markers = ["HELP_PEOPLE", "HELP_PEOPLE", "HELP_ENVIRONMENT", "HELP_ENVIRONMENT", "TANGIBLE_BENEFITS", "TANGIBLE_BENEFITS", "SPIRIT_OF_EXPLORATION", "SPIRIT_OF_EXPLORATION"]

    for line in lines:
        for i in range(len(match_examples)):
            color = colors[color_marker_mapper[i]]
            marker = markers[i]
            phrase_list = match_examples[i]
            phrase_set = set()
            for j in range(len(phrase_list)):
                phrase = " ".join(phrase_list[j])
                phrase_set.add(phrase)
            for phrase in phrase_set:
                if phrase in mapper:
                    full_phrase = mapper[phrase]
                    line = line.replace(full_phrase, f'<span style="background-color:{color}">{full_phrase}<sub>[{marker}]</sub></span>')
        new_lines.append(line)

    with open(os.path.join(settings.BASE_DIR, "templates/space_essay_color.html"), "w") as f:
        f.writelines(new_lines)


def write_color_input_html(submited_essay, input_topic_mapper):
    topic_color_mapper = {"HOSPITALS": "#00FF00", "MALARIA": "yellow", "FARMING": "#FF00FF", "SCHOOL": "#00CDFF"}
    for key, val in input_topic_mapper.items():
        color = topic_color_mapper[val]
        submited_essay = submited_essay.replace(key, f"<span style='background-color:{color}'>{key}<sub>[TOPIC:{val}]</sub></span>")
    with open(os.path.join(settings.BASE_DIR, "templates/mvp_submission_color.html"), "w") as f:
        f.write(submited_essay)


def write_color_input_html_space(submited_essay, input_topic_mapper):
    topic_color_mapper = {"HELP_PEOPLE": "#00FF00", "HELP_ENVIRONMENT": "yellow", "TANGIBLE_BENEFITS": "#FF00FF", "SPIRIT_OF_EXPLORATION": "#00CDFF"}
    for key, val in input_topic_mapper.items():
        color = topic_color_mapper[val]
        submited_essay = submited_essay.replace(key, f"<span style='background-color:{color}'>{key}<sub>[TOPIC:{val}]</sub></span>")
    with open(os.path.join(settings.BASE_DIR, "templates/space_submission_color.html"), "w") as f:
        f.write(submited_essay)



def hello(request):
    return HttpResponse("Hello world ! ")


def index(request):
    views_dict = {"name": "eRevise"}
    return render(request, "index.html", {"views_dict": views_dict})




def add_data_user(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')
    # now = datetime.datetime.now()
    now = timezone.now()
    user = User(
        user_name='zliu',
        password="123",
        first_name="Zhexiong",
        last_name="Liu",
        permission=1,
        created_time=now.strftime('%Y-%m-%d %H:%M:%S'),
        modified_time=now.strftime('%Y-%m-%d %H:%M:%S')
    )
    user.save()
    return HttpResponse("<p>user added </p>")


def add_data_prompt(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')
    # now = datetime.datetime.now()
    now = timezone.now()
    items = (("MVP", 'The author described how the quality of life was improved by the Millennium Villages Project in Sauri,          Kenya. Based on the article, did the author convince you that "winning the fight against poverty is          achievable in our lifetime"? Explain why or why not with 3-4 examples from the text to support your answer.'),
             ("Space", '         Consider the reasons given in the article for why we should and should not fund space exploration. Did the author convince you that "space exploration is desirable when there is so much that needs to be done on earth"? Give reasons for your answer. Support your reasons with 3-4 pieces of evidence from the text.         '))
    for item in items:
        prompt = Prompt(
            prompt_name=item[0],
            prompt_content=item[1],
            threshold=0.7,
            created_time=now.strftime('%Y-%m-%d %H:%M:%S'),
            modified_time=now.strftime('%Y-%m-%d %H:%M:%S')
        )
        prompt.save()
    return HttpResponse("<p>prompt added </p>")


def add_data_essay(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')
    # now = datetime.datetime.now()
    now = timezone.now()
    essay = Essay(
        user_name='zliu',
        prompt_name='MVP',
        essay_content="I like the essay and want to take a look",
        essay_revision=0,
        submitted=False,
        processed=False,
        submitted_time=now,
        processed_time=now,
        created_time=now,
        modified_time=now
    )
    essay.save()
    return HttpResponse("<p>essay added </p>")

def add_data_feedback(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')
    # now = datetime.datetime.now()
    now = timezone.now()
    itmes = (('MVP', 'Use more evidence from the article', '<ul style="padding:15px;">\n<li>Re-read the article and the writing prompt.</li>\n<li>Choose at least three different pieces of evidence to support your argument.</li>\n<li>Consider the whole article as you select your evidence.</li>\n</ul>'),
    ('MVP','Provide more details for each piece of evidence you use', '<ul style="padding:15px;">\n<li>Add more specific details about each piece of evidence.</li>\n<ul>\n\t<li>For example, writing, "The school fee was a problem" is not specific enough. It is better to write, "Students could not attend school because they did not have enough money to pay the school fee."</li>\n</ul>\n<li>Use your own words to describe the evidence.</li>\n</ul>'),
    ('MVP','Explain the evidence', '<ul style="padding:15px;">\n<li>Tell your reader why you included each piece of evidence. Explain how the evidence helps to make your point.</li>\n</ul>'),
    ('MVP','Explain how the evidence connects to the main idea & elaborate', '<ul style="padding:15px;">\n<li>Tie the evidence not only to the point you are making within a paragraph, but to your overall argument.</li>\n<li>Elaborate. Give a detailed and clear explanation of how the evidence supports your argument.</li>\n</ul>'),
    ('Space','Use more evidence from the article', '<ul style="padding:15px;">\n<li>Re-read the article and the writing prompt.</li>\n<li>Choose at least three different pieces of evidence to support your argument.</li>\n<li>Consider the whole article as you select your evidence.</li>\n</ul>'),
    ('Space','Provide more details for each piece of evidence you use', '<ul style="padding:15px;">\n<li>Add more specific details about each piece of evidence.</li>\n<ul>\n\t<li>For example, writing, "Funding space exploration made people healthier" is not specific enough. It is better to write, "Funding space exploration helped develop medical instruments that taught us about the body’s reaction to stress."</li>\n</ul>\n<li>Use your own words to describe the evidence.</li>\n</ul>'),
    ('Space','Explain the evidence', '<ul style="padding:15px;">\n<li>Tell your reader why you included each piece of evidence. Explain how the evidence helps to make your point.</li>\n</ul>'),
    ('Space','Explain how the evidence connects to the main idea & elaborate', '<ul style="padding:15px;">\n<li>Tie the evidence not only to the point you are making within a paragraph, but to your overall argument.</li>\n<li>Elaborate. Give a detailed and clear explanation of how the evidence supports your argument.</li>\n</ul>'))

    for i in range(len(itmes)):
        item = itmes[i]
        feedback = Feedback(
            prompt_name=item[0],
            level=int((i)%4)+1,
            title=item[1],
            content=item[2],
            created_time=now,
            modified_time=now
        )
        feedback.save()
    return HttpResponse("<p>feedback added </p>")


def get_data_prompt(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')
    response = ""
    fields = Prompt.objects.all()
    for field in fields:
        response += f"{field.prompt_name}, content {field.prompt_content} "
    return HttpResponse("<p>" + response + "</p>")


def get_data_essay(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')
    response = ""
    fields = Essay.objects.all()
    for field in fields:
        response += f"{field.user_name}, {field.prompt_name}, content {field.essay_content} "
    return HttpResponse("<p>" + response + "</p>")

def upload_file(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            df = pd.read_excel(file)

            # queryset = Classroom.objects.all()
            # exist_df = pd.DataFrame.from_records(queryset.values())

            # create student account
            for _, row in df.iterrows():
                student_username = f"{row['student_firstname'].lower()[:2].capitalize()}{row['student_lastname'].lower()[:2].capitalize()}"
                Classroom.objects.update_or_create(
                    teacher_id=row["teacher_id"],
                    student_id=row["student_id"],
                    defaults={"teacher_firstname":row['teacher_firstname'],
                    "teacher_lastname":row['teacher_lastname'],
                    "student_username":student_username}
                )
                username = str(row["teacher_id"])+str(row["student_id"])
                # new_user = User(username=username, password=username)

                existing_user = User.objects.filter(username=username).first()
                if existing_user:
                    existing_user.delete()

                # User doesn't exist, create a new one
                new_user = User.objects.create_user(username=username, password=username, first_name=f"{student_username} ({row['teacher_id']}{row['student_id']})")
                new_user.save()



            messages.success(request, 'File uploaded successfully!')

            return redirect('upload_file')
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})


def upload_file_v2(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            df = pd.read_excel(file)

            # queryset = Classroom.objects.all()
            # exist_df = pd.DataFrame.from_records(queryset.values())

            for _, row in df.iterrows():
                teacher_firstname = row["teacher_firstname"].strip()
                teacher_lastname = row["teacher_lastname"].strip()
                teacher_id = row["teacher_id"]
                student_firstname = row["student_firstname"].strip()
                student_lastname = row["student_lastname"].strip()
                student_id = row["student_id"]

                essay_id = str(student_id)[1]

                student_name = f"{student_firstname.lower()[:2].capitalize()}{student_lastname.lower()[:2].capitalize()}"
                teacher_name = f"{teacher_firstname.lower().capitalize()}"

                # create class
                Classroom.objects.update_or_create(
                    teacher_id=teacher_id,
                    student_id=student_id,
                    defaults={"teacher_name":teacher_name,
                    "student_name":student_name}
                )
                student_username = str(teacher_id)+str(student_id)
                # new_user = User(username=username, password=username)

                existing_user = User.objects.filter(username=student_username).first()
                if existing_user:
                    existing_user.delete()
                    # continue

                # User doesn't exist, create a new one
                new_user = User.objects.create_user(username=student_username, password=student_username, first_name=f"{student_name} ({teacher_id}{student_id})", email=essay_id)
                new_user.save()

                # Save the user to the database
                # new_user.save()

            # create teacher account
            for _, row in df.iterrows():
                username = str(row["teacher_id"])
                # new_user = User(username=username, password=username)

                existing_user = User.objects.filter(username=username).first()
                if existing_user:
                    existing_user.delete()

                # User doesn't exist, create a new one
                firstname = row['teacher_firstname'].lower().capitalize().strip()
                lastname = row['teacher_lastname'].lower().capitalize().strip()
                student_id = row["student_id"]
                essay_id = str(student_id)[1]
                teacher_name = firstname
                zipcode = str(row['zipcode'])
                # zipcode = str(15260)
                token = "!"
                password = lastname + zipcode + token
                new_user = User.objects.create_user(username=username, password=password,
                                                    first_name=f"{teacher_name} ({row['teacher_id']})", email=essay_id, is_staff=True)
                new_user.save()


            messages.success(request, 'File uploaded successfully!')

            return redirect('upload_file')
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})

def get_user_info(request):

    teacher_info = None
    student_info = None

    if request.method == 'POST':
        form = UsernameForm(request.POST)
        if form.is_valid():
            teacher_id = form.cleaned_data['teacher_id']
            try:
                teacher_info = Classroom.objects.filter(teacher_id=teacher_id)
                if len(teacher_info) > 0:
                    teacher_info = teacher_info[0]
                else:
                    teacher_info = None
                    messages.error(request, 'Wrong teacher information!')
            except Classroom.DoesNotExist:
                teacher_info = None
                messages.error(request, 'Wrong teacher information!')

            if teacher_info:
                student_id = form.cleaned_data['student_id']
                try:
                    student_info = Classroom.objects.filter(Q(student_id=student_id) & Q(teacher_id=teacher_id))
                    if len(student_info) > 0:
                        student_info = student_info[0]
                    else:
                        student_info = None
                        messages.error(request, 'Wrong student information!')
                except Classroom.DoesNotExist:
                    student_info = None
                    messages.error(request, 'Wrong student information!')

            if teacher_info and student_info:
                username = str(teacher_id)+str(student_id)
                user = authenticate(username=username, password=username)
                if user is not None:
                    login(request, user)
    else:
        form = UsernameForm()

    return render(request, 'confirm.html', {'form': form, 'teacher_info': teacher_info, 'student_info': student_info})


def roster(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')


    if request.user.is_superuser:
        people = Classroom.objects.all()
    elif request.user.is_staff:
        people = Classroom.objects.filter(teacher_id=request.user.username).all()
    else:
        return render(request, 'index.html')


    # print(people)
    context = {
        'people': people,
    }
    return render(request, 'roster.html', context)

def submission(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')
    # submission = Essay.objects.all()
    if request.user.is_superuser:
        submission = Essay.objects.values('user_name', 'essay_revision', 'prompt_name').distinct()
    elif request.user.is_staff:
        submission = Essay.objects.filter(user_name__startswith=request.user.username).values('user_name', 'essay_revision', 'prompt_name').distinct()
    else:
        return render(request, 'index.html')
    latest_records = []
    for distinct_value in submission:
        # Retrieve the latest record for each distinct value in the 'field_name' field
        latest_record = Essay.objects.filter(user_name=distinct_value['user_name'],
                                                 essay_revision=distinct_value['essay_revision'],
                                                 prompt_name=distinct_value['prompt_name']).order_by(
            'submitted_time').last()
        if latest_record:
            if latest_record.processed:
                processed_entry = Process.objects.filter(user_name=distinct_value['user_name'],
                                     essay_revision=distinct_value['essay_revision'],
                                     prompt_name=distinct_value['prompt_name']).order_by(
                    'processed_time').last()
                auto_feedback = processed_entry.auto_feedback
                teacher_feedback = processed_entry.teacher_feedback

                soup = BeautifulSoup(auto_feedback, 'html.parser')
                auto_feedback = soup.get_text(separator='\n', strip=True)

            else:
                auto_feedback = ""
                teacher_feedback = ""

            latest_records.append([latest_record, auto_feedback, teacher_feedback])
    # print(people)
    context = {
        'submission': latest_records,
    }
    return render(request, 'submission.html', context)


def save_feedback(request):
    if not request.user.is_authenticated:
        return render(request, 'index.html')
    if request.method == 'POST':
        try:
            # Parse request body
            data = json.loads(request.body)
            user_name = data.get('user_name')
            essay_revision = data.get('essay_revision')
            teacher_feedback = data.get('feedback')
            prompt_name = data.get('prompt_name')

            # Update feedback in the database
            process_entry = Process.objects.get(user_name=user_name, prompt_name=prompt_name, essay_revision=essay_revision)
            process_entry.teacher_feedback = teacher_feedback
            process_entry.save()

            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    else:
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
