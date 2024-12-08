# Generated by Django 4.1.2 on 2024-04-11 04:45

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Classroom',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('teacher_id', models.CharField(max_length=255, unique=False)),
                ('teacher_name', models.CharField(max_length=255)),
                ('student_id', models.CharField(max_length=255, unique=False)),
                ('student_name', models.CharField(max_length=255)),
            ],
            options={
                'db_table': 'classroom',
            },
        ),
        # migrations.CreateModel(
        #     name='Essay',
        #     fields=[
        #         ('id', models.AutoField(primary_key=True, serialize=False)),
        #         ('user_name', models.CharField(max_length=255, verbose_name='user_name')),
        #         ('prompt_name', models.CharField(max_length=255, verbose_name='prompt_name')),
        #         ('essay_content', models.TextField(verbose_name='essay_content')),
        #         ('essay_revision', models.IntegerField(default=0, verbose_name='essay_revision')),
        #         ('submitted', models.BooleanField(default=False, verbose_name='submitted')),
        #         ('processed', models.BooleanField(default=False, verbose_name='processed')),
        #         ('submitted_time', models.DateTimeField(verbose_name='submitted_time')),
        #         ('processed_time', models.DateTimeField(verbose_name='processed_time')),
        #         ('created_time', models.DateTimeField(verbose_name='created_time')),
        #         ('modified_time', models.DateTimeField(verbose_name='modified_time')),
        #     ],
        #     options={
        #         'db_table': 'essay',
        #     },
        # ),
        migrations.CreateModel(
            name='Feedback',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('prompt_name', models.CharField(max_length=255, verbose_name='prompt_name')),
                ('level', models.IntegerField(verbose_name='level')),
                ('title', models.TextField(verbose_name='title')),
                ('content', models.TextField(verbose_name='content')),
                ('created_time', models.DateTimeField(verbose_name='created_time')),
                ('modified_time', models.DateTimeField(verbose_name='modified_time')),
            ],
            options={
                'db_table': 'feedback',
            },
        ),
        migrations.CreateModel(
            name='Process',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('user_name', models.CharField(max_length=255, verbose_name='user_name')),
                ('prompt_name', models.CharField(max_length=255, verbose_name='prompt_name')),
                ('essay_revision', models.IntegerField(default=0, verbose_name='essay_revision')),
                ('essay_content', models.TextField(verbose_name='essay_content')),
                ('npe_score', models.IntegerField(default=0, verbose_name='npe_score')),
                ('spc_score', models.IntegerField(default=0, verbose_name='spc_score')),
                ('npe_keyword', models.TextField(verbose_name='npe_keyword')),
                ('spc_keyword', models.TextField(verbose_name='spc_keyword')),
                ('feedback_level', models.FloatField(verbose_name='feedback_level')),
                ('auto_feedback', models.TextField(verbose_name='auto_feedback')),
                ('teacher_feedback', models.TextField(null=True, verbose_name='teacher_feedback')),
                ('annotating_line_number', models.TextField(null=True, verbose_name='annotating_line_number')),
                ('annotating_text', models.TextField(null=True, verbose_name='annotating_text')),
                ('processed_time', models.DateTimeField(verbose_name='processed_time')),
            ],
            options={
                'db_table': 'process',
            },
        ),
        migrations.CreateModel(
            name='ProcessRevision',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('user_name', models.CharField(max_length=255, verbose_name='user_name')),
                ('prompt_name', models.CharField(max_length=255, verbose_name='prompt_name')),
                ('essay_revision', models.IntegerField(default=0, verbose_name='essay_revision')),
                ('old_essay', models.TextField(null=True, verbose_name='old_essay')),
                ('new_essay', models.TextField(null=True, verbose_name='new_essay')),
                ('old_sentence_id', models.CharField(max_length=255, null=True, verbose_name='old_sentence_id')),
                ('old_sentence_aligned_id', models.CharField(max_length=255, null=True, verbose_name='old_sentence_aligned_id')),
                ('old_sentence', models.TextField(null=True, verbose_name='old_sentence')),
                ('new_sentence', models.TextField(null=True, verbose_name='new_sentence')),
                ('new_sentence_id', models.CharField(max_length=255, null=True, verbose_name='new_sentence_id')),
                ('new_sentence_aligned_id', models.CharField(max_length=255, null=True, verbose_name='new_sentence_aligned_id')),
                ('old_argument_context', models.TextField(null=True, verbose_name='old_argument_context')),
                ('new_argument_context', models.TextField(null=True, verbose_name='new_argument_context')),
                ('used_context', models.TextField(null=True, verbose_name='used_context')),
                ('used_sentence', models.TextField(null=True, verbose_name='used_sentence')),
                ('coarse_label', models.CharField(max_length=255, null=True, verbose_name='coarse_label')),
                ('fine_label', models.CharField(max_length=255, null=True, verbose_name='fine_label')),
                ('successfulness', models.CharField(max_length=255, null=True, verbose_name='successfulness')),
                ('mentioned_topic', models.CharField(max_length=255, null=True, verbose_name='mentioned_topic')),
                ('old_npe_score', models.IntegerField(verbose_name='old_npe_score')),
                ('new_npe_score', models.IntegerField(verbose_name='new_npe_score')),
                ('old_spc_score', models.IntegerField(verbose_name='old_spc_score')),
                ('new_spc_score', models.IntegerField(verbose_name='new_spc_score')),
                ('old_npe_keyword', models.TextField(verbose_name='old_npe_keyword')),
                ('new_npe_keyword', models.TextField(verbose_name='new_npe_keyword')),
                ('old_spc_keyword', models.TextField(verbose_name='old_spc_keyword')),
                ('new_spc_keyword', models.TextField(verbose_name='new_spc_keyword')),
                ('auto_feedback', models.TextField(verbose_name='auto_feedback')),
                ('evidence_feedback_level', models.FloatField(verbose_name='evidence_feedback_level')),
                ('revision_feedback_level', models.FloatField(verbose_name='revision_feedback_level')),
                ('annotating_text', models.TextField(null=True, verbose_name='annotating_text')),
                ('annotating_line_number', models.TextField(null=True, verbose_name='annotating_line_number')),
                ('processed_time', models.DateTimeField(verbose_name='processed_time')),
            ],
            options={
                'db_table': 'process_revision',
            },
        ),
        migrations.CreateModel(
            name='Prompt',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('prompt_name', models.CharField(max_length=255, verbose_name='prompt_name')),
                ('prompt_content', models.TextField(default='', verbose_name='prompt_content')),
                ('threshold', models.FloatField(verbose_name='threshold')),
                ('created_time', models.DateTimeField(verbose_name='created_time')),
                ('modified_time', models.DateTimeField(verbose_name='modified_time')),
            ],
            options={
                'db_table': 'prompt',
            },
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('user_name', models.CharField(max_length=255, verbose_name='user_name')),
                ('password', models.CharField(max_length=255, verbose_name='password')),
                ('first_name', models.CharField(max_length=255, verbose_name='first_name')),
                ('last_name', models.CharField(max_length=255, verbose_name='last_name')),
                ('permission', models.IntegerField(default=1, verbose_name='permission')),
                ('created_time', models.DateTimeField(verbose_name='created_time')),
                ('modified_time', models.DateTimeField(verbose_name='modified_time')),
            ],
            options={
                'db_table': 'user',
            },
        ),
        migrations.AddIndex(
            model_name='user',
            index=models.Index(fields=['user_name'], name='user_user_na_ad4ee2_idx'),
        ),
        migrations.AddIndex(
            model_name='prompt',
            index=models.Index(fields=['prompt_name'], name='prompt_prompt__fbe4af_idx'),
        ),
        migrations.AddIndex(
            model_name='processrevision',
            index=models.Index(fields=['user_name', 'prompt_name', 'essay_revision'], name='process_rev_user_na_40abc7_idx'),
        ),
        migrations.AddIndex(
            model_name='process',
            index=models.Index(fields=['user_name', 'prompt_name', 'essay_revision'], name='process_user_na_114aee_idx'),
        ),
        migrations.AlterUniqueTogether(
            name='feedback',
            unique_together={('prompt_name', 'level')},
        ),
        # migrations.AddIndex(
        #     model_name='essay',
        #     index=models.Index(fields=['user_name', 'prompt_name', 'essay_revision'], name='essay_user_na_09d69c_idx'),
        # ),
        migrations.AddIndex(
            model_name='classroom',
            index=models.Index(fields=['id'], name='id_7f34cb_idx'),
        ),
    ]
