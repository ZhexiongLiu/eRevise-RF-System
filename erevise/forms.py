from django.contrib.auth import authenticate
from django.contrib import auth
from django import forms
from database.models import User, Essay, Prompt

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class CustomUserCreationForm(UserCreationForm):
    custom_field = forms.CharField(max_length=100)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = UserCreationForm.Meta.fields + ('custom_field',)

class UsersLoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput, )

    def __init__(self, *args, **kwargs):
        super(UsersLoginForm, self).__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({
            'class': 'form-control',
            "name": "username"})
        self.fields['password'].widget.attrs.update({
            'class': 'form-control',
            "name": "password"})

    def clean(self, *args, **keyargs):
        username = self.cleaned_data.get("username")
        password = self.cleaned_data.get("password")

        if username and password:
            user = authenticate(username=username, password=password)
            if not user:
                raise forms.ValidationError("This user name or password does not match!")
            if not user.check_password(password):
                raise forms.ValidationError("Incorrect Password")
            if not user.is_active:
                raise forms.ValidationError("User is no longer active")

        return super(UsersLoginForm, self).clean(*args, **keyargs)




class RevisionForm(forms.Form):
    REVISION_PHASE = {(3, "Version 3"), (2, "Version 2"), (1, "Version 1"), }
    essay_revision = forms.ChoiceField(widget=forms.Select(attrs={'onchange': 'this.form.submit();'}), choices=REVISION_PHASE, initial=0, required=False)
    # submited_essay = forms.CharField(widget=forms.Textarea(), required=False)

    def __init__(self, *args, **kwargs):
        super(RevisionForm, self).__init__(*args, **kwargs)
        self.fields['essay_revision'].widget.attrs.update({
            # 'class': 'form-control',
            "name": "essay_revision"})

        # self.fields['submited_essay'].widget.attrs.update({
        #     'class': 'form-control',
        #     "name": "submited_essay",
        #     "row": 3})


    class Meta:
        model = Essay
        fields = "__all__"


class EssayForm(forms.Form):


    # revision_phase = forms.ChoiceField(widget=forms.Select(), choices=Prompt.objects.values_list("id", "prompt_name"))
    # essay_revision = forms.ChoiceField(widget=forms.Select(), choices=REVISION_PHASE, initial=1)

    essay = forms.CharField(widget=forms.Textarea(attrs={'id': 'essay_text_box'}), required=False)

    def __init__(self, *args, **kwargs):
        super(EssayForm, self).__init__(*args, **kwargs)
        # self.fields['essayname'].widget.choices = Prompt.objects.values_list("id", "prompt_name")

        # self.fields['essay_revision'].widget.attrs.update({
        #     'class': 'form-control',
        #     "name": "essay_revision"})

        self.fields['essay'].widget.attrs.update({
            'class': 'form-control',
            "name": "essay",
            "row":5})
        # self.fields['prompt'].widget = (Prompt.objects.values_list("prompt_name"))

        # self.fields['username'].widget.attrs.update({
        #     'class': 'form-control',
        #     "name": "username"})
        # self.fields['password'].widget.attrs.update({
        #     'class': 'form-control',
        #     "name": "password"})

    # def clean_essay_revision(self):
    #     essay_revision = self.cleaned_data['essay_revision']
    #     if essay_revision is None:
    #         raise forms.ValidationError("Your value is none!")
    #
    #     return essay_revision

    # def get_essay_revision(self):
    #     return self.cleaned_data["essay_revision"]

    # def clean(self, *args, **keyargs):
    #     username = self.cleaned_data.get("username")
    #     password = self.cleaned_data.get("password")
    #
    #     if username and password:
    #         user = authenticate(username=username, password=password)
    #         if not user:
    #             raise forms.ValidationError("This user name or password does not match!")
    #         if not user.check_password(password):
    #             raise forms.ValidationError("Incorrect Password")
    #         if not user.is_active:
    #             raise forms.ValidationError("User is no longer active")
    #
    #     return super(EssayForm, self).clean(*args, **keyargs)

    class Meta:
        model = Essay
        fields = "__all__"

class UploadFileForm(forms.Form):
    file = forms.FileField(label='')

class UsernameForm(forms.Form):
    teacher_id = forms.CharField(label='', max_length=255)
    student_id = forms.CharField(label='', max_length=255)