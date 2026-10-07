from flask import Blueprint, render_template, request, current_app
import numpy as np

heart_bp = Blueprint('heart', __name__)


@heart_bp.route('/heart')
def heart():
    return render_template('heart.html')


@heart_bp.route('/heartpred', methods=['POST'])
def heartpred():
    try:
        fields = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
        values = [float(request.form[field]) for field in fields]
        input_array = np.asarray(values).reshape(1, -1)

        prediction = current_app.heart_model.predict(input_array)
        result = (
            "The person does not have a Heart Disease"
            if prediction[0] == 0
            else "The person has Heart Disease"
        )
    except (ValueError, KeyError):
        result = "Invalid input. Please enter numeric values in all fields."

    return render_template('prediction.html', pre=result)
