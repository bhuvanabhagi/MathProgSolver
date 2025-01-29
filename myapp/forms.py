from django import forms

class LPForm(forms.Form):
    objective_function = forms.CharField(label='Objective Function', max_length=100)
    constraints = forms.CharField(label='Constraints', widget=forms.Textarea)
