from django.db import models

class User(models.Model):
    id = models.AutoField(primary_key=True)
    user_name = models.CharField("user_name", max_length=255)
    password = models.CharField("password", max_length=255)
    first_name = models.CharField("first_name", max_length=255)
    last_name = models.CharField("last_name", max_length=255)
    permission = models.IntegerField("permission", default=1)
    created_time = models.DateTimeField("created_time")
    modified_time = models.DateTimeField("modified_time")

    class Meta:
        db_table = "user"
        indexes = [
            models.Index(fields=['user_name'], )
        ]

class Prompt(models.Model):
    id = models.AutoField(primary_key=True)
    prompt_name = models.CharField("prompt_name", max_length=255)
    prompt_content = models.TextField("prompt_content", default="")
    threshold = models.FloatField("threshold")
    created_time = models.DateTimeField("created_time")
    modified_time = models.DateTimeField("modified_time")

    class Meta:
        db_table = "prompt"
        indexes = [
            models.Index(fields=['prompt_name'])
        ]

class Essay(models.Model):
    id = models.AutoField(primary_key=True)
    user_name = models.CharField("user_name", null=False, max_length=255)
    prompt_name = models.CharField("prompt_name", null=False, max_length=255)
    essay_content = models.TextField("essay_content", null=False)
    essay_revision = models.IntegerField("essay_revision", default=0)
    submitted = models.BooleanField("submitted", default=False)
    processed = models.BooleanField("processed", default=False)
    submitted_time = models.DateTimeField("submitted_time")
    processed_time = models.DateTimeField("processed_time")
    created_time = models.DateTimeField("created_time")
    modified_time = models.DateTimeField("modified_time")

    class Meta:
        db_table = "essay"
        # unique_together = (("user_name", "prompt_name", "essay_revision"),)
        indexes = [models.Index(fields=['user_name', 'prompt_name', 'essay_revision'])]


class Process(models.Model):
    id = models.AutoField(primary_key=True)
    user_name = models.CharField("user_name", null=False, max_length=255)
    prompt_name = models.CharField("prompt_name", null=False, max_length=255)
    essay_revision = models.IntegerField("essay_revision", default=0)
    essay_content = models.TextField("essay_content", null=False)
    npe_score = models.IntegerField("npe_score", default=0)
    spc_score = models.IntegerField("spc_score", default=0)
    npe_keyword = models.TextField("npe_keyword", null=False)
    spc_keyword = models.TextField("spc_keyword", null=False)
    feedback_level = models.FloatField("feedback_level", null=False)
    auto_feedback = models.TextField("auto_feedback", null=False)
    teacher_feedback = models.TextField("teacher_feedback", null=True)
    annotating_line_number = models.TextField("annotating_line_number", null=True)
    annotating_text = models.TextField("annotating_text", null=True)
    processed_time = models.DateTimeField("processed_time")

    class Meta:
        db_table = "process"
        # unique_together = (("user_name", "prompt_name", "essay_revision"),)
        indexes = [models.Index(fields=['user_name', 'prompt_name', 'essay_revision'])]

    def __str__(self):
        return self.user_name

class ProcessRevision(models.Model):
    id = models.AutoField(primary_key=True)
    user_name = models.CharField("user_name", null=False, max_length=255)
    prompt_name = models.CharField("prompt_name", null=False, max_length=255)
    essay_revision = models.IntegerField("essay_revision", default=0, null=False)
    old_essay = models.TextField("old_essay", null=True)
    new_essay = models.TextField("new_essay", null=True)
    old_sentence_id = models.CharField("old_sentence_id", null=True, max_length=255)
    old_sentence_aligned_id = models.CharField("old_sentence_aligned_id", null=True, max_length=255)
    old_sentence = models.TextField("old_sentence", null=True)
    new_sentence = models.TextField("new_sentence", null=True)
    new_sentence_id = models.CharField("new_sentence_id", null=True, max_length=255)
    new_sentence_aligned_id = models.CharField("new_sentence_aligned_id", null=True, max_length=255)
    old_argument_context = models.TextField("old_argument_context", null=True)
    new_argument_context = models.TextField("new_argument_context", null=True)
    used_context = models.TextField("used_context", null=True)
    used_sentence = models.TextField("used_sentence", null=True)
    coarse_label = models.CharField("coarse_label", null=True, max_length=255)
    fine_label = models.CharField("fine_label", null=True,  max_length=255)
    successfulness = models.CharField("successfulness", null=True, max_length=255)
    mentioned_topic = models.CharField("mentioned_topic", null=True,  max_length=255)
    old_npe_score = models.IntegerField("old_npe_score", null=False)
    new_npe_score = models.IntegerField("new_npe_score", null=False)
    old_spc_score = models.IntegerField("old_spc_score", null=False)
    new_spc_score = models.IntegerField("new_spc_score", null=False)
    old_npe_keyword = models.TextField("old_npe_keyword", null=False)
    new_npe_keyword = models.TextField("new_npe_keyword", null=False)
    old_spc_keyword = models.TextField("old_spc_keyword", null=False)
    new_spc_keyword = models.TextField("new_spc_keyword", null=False)
    auto_feedback = models.TextField("auto_feedback", null=False)
    evidence_feedback_level = models.FloatField("evidence_feedback_level", null=False)
    revision_feedback_level = models.FloatField("revision_feedback_level", null=False)
    annotating_text = models.TextField("annotating_text", null=True)
    annotating_line_number = models.TextField("annotating_line_number", null=True)
    processed_time = models.DateTimeField("processed_time")

    class Meta:
        db_table = "process_revision"
        # unique_together = (("user_name", "prompt_name", "essay_revision"),)
        indexes = [models.Index(fields=['user_name', 'prompt_name', 'essay_revision'])]

    def __str__(self):
        return self.user_name


class Feedback(models.Model):
    id = models.AutoField(primary_key=True)
    prompt_name = models.CharField("prompt_name", max_length=255)
    level = models.IntegerField("level", null=False)
    title = models.TextField("title", null=False)
    content = models.TextField("content", null=False)
    created_time = models.DateTimeField("created_time")
    modified_time = models.DateTimeField("modified_time")

    class Meta:
        db_table = "feedback"
        unique_together = (("prompt_name", "level"),)
        # indexes = [models.Index(fields=['prompt_name', 'level'])]



class Classroom(models.Model):
    teacher_id = models.CharField(max_length=255, unique=False)
    teacher_name = models.CharField(max_length=255)
    # teacher_class = models.CharField(max_length=255)
    student_id = models.CharField(max_length=255, unique=False)
    student_name = models.CharField(max_length=255)

    class Meta:
        db_table = "classroom"
        indexes = [
            models.Index(fields=['id'], )
        ]






