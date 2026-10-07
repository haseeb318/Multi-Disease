from flask import Blueprint, render_template, request, current_app
import numpy as np

diabetes_bp = Blueprint('diabetes', __name__)


@diabetes_bp.route('/diabete')
def diabete():
    return render_template('diabetes.html')


@diabetes_bp.route('/pred', methods=['POST'])
def pred():
    try:
        fields = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        values = [float(request.form[field]) for field in fields]
        input_array = np.asarray(values).reshape(1, -1)

        prediction = current_app.diabetes_model.predict(input_array)
        result = (
            "The person is not diabetic"
            if prediction[0] == 0
            else "The person is diabetic"
        )
    except (ValueError, KeyError):
        result = "Invalid input. Please enter numeric values in all fields."

    return render_template('prediction.html', pre=result)
