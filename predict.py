import os
import sys
import argparse
import pickle
import joblib
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

CONTINUOUS = ['age', 'avg_glucose_level', 'bmi']

COLUMNS = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'avg_glucose_level', 'bmi',
    'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
    'work_type_Self-employed', 'work_type_children',
    'Residence_type_Rural', 'Residence_type_Urban',
    'smoking_status_Unknown', 'smoking_status_formerly smoked',
    'smoking_status_never smoked', 'smoking_status_smokes',
]


def build_row(age, hypertension, heart_disease, ever_married,
              work_type, residence_type, avg_glucose_level,
              bmi, smoking_status, gender):
    row = {
        'gender': 1 if gender == 'Male' else 0,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': 1 if ever_married == 'Yes' else 0,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
    }
    for wt in ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children']:
        row[f'work_type_{wt}'] = 1 if work_type == wt else 0
    for rt in ['Rural', 'Urban']:
        row[f'Residence_type_{rt}'] = 1 if residence_type == rt else 0
    for ss in ['Unknown', 'formerly smoked', 'never smoked', 'smokes']:
        row[f'smoking_status_{ss}'] = 1 if smoking_status == ss else 0
    return row


def build_scaler():
    # Parametros del StandardScaler calculados sobre el dataset de entrenamiento.
    # Orden: [age, avg_glucose_level, bmi]
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.mean_  = np.array([53.90506438, 119.86899127,  30.0791309 ])
    sc.scale_ = np.array([22.18603598,  55.29653441,   7.54602733])
    sc.var_   = sc.scale_ ** 2
    sc.n_features_in_ = 3
    sc.feature_names_in_ = np.array(CONTINUOUS)
    return sc


def load_models(model_dir):
    models = {}
    models['scaler'] = build_scaler()

    rfi_path = os.path.join(model_dir, 'rf_improved_model.joblib')
    MLP_path = os.path.join(model_dir, 'MLP_deep.keras')

    if os.path.exists(rfi_path):
        models['rf_improved_model'] = joblib.load(rfi_path)

    if os.path.exists(MLP_path):
        models['MLP_deep'] = tf.keras.models.load_model(MLP_path)

    return models


def predict(models, row_dict):
    X = pd.DataFrame([row_dict]).reindex(columns=COLUMNS, fill_value=0)
    X = X.astype(np.float32)
    X[CONTINUOUS] = models['scaler'].transform(X[CONTINUOUS])

    results = []

    if 'rf_improved_model' in models:
        prob = models['rf_improved_model'].predict_proba(X)[0][1]
        results.append(('Random Forest', prob))

    if 'MLP_deep' in models:
        prob = float(models['MLP_deep'].predict(X.values, verbose=0)[0][0])
        results.append(('MLP deep', prob))

    return results


def print_results(patient_info, results):
    print('=' * 58)
    print(f"Paciente: {patient_info['age']}a, "
          f"glucosa={patient_info['avg_glucose_level']}, "
          f"bmi={patient_info['bmi']}")
    print('-' * 58)
    for name, prob in results:
        label = 'STROKE' if prob >= 0.5 else 'No Stroke'
        bar = chr(9608) * int(prob * 20)
        print(f'  {name:25s}: {prob:.3f} {bar:<20} {label}')
    print('=' * 58)


def random_patient():
    age = round(random.uniform(1, 82), 1)
    gender = random.choice(['Male', 'Female'])
    hypertension = random.choices([0, 1], weights=[90, 10])[0]
    heart_disease = random.choices([0, 1], weights=[94, 6])[0]
    ever_married = random.choice(['Yes', 'No'])
    work_type = random.choice(['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'])
    residence_type = random.choice(['Rural', 'Urban'])
    avg_glucose_level = round(random.uniform(55, 270), 1)
    bmi = round(random.uniform(10, 60), 1)
    smoking_status = random.choice(['Unknown', 'formerly smoked', 'never smoked', 'smokes'])
    return dict(
        age=age, gender=gender, hypertension=hypertension,
        heart_disease=heart_disease, ever_married=ever_married,
        work_type=work_type, residence_type=residence_type,
        avg_glucose_level=avg_glucose_level, bmi=bmi,
        smoking_status=smoking_status
    )


def ask(prompt, options=None, cast=str, default=None):
    hint = ''
    if default is not None:
        hint = f' [Enter = {default}]'
    if options:
        hint += f' ({"/".join(options)})'
    while True:
        raw = input(f'{prompt}{hint}: ').strip()
        if raw == '' and default is not None:
            return default
        try:
            val = cast(raw)
            if options and val not in options:
                print(f'  Opciones validas: {", ".join(options)}')
                continue
            return val
        except (ValueError, TypeError):
            print(f'  Valor no valido.')


def interactive_mode(models):
    print('\nModo interactivo. Presiona Ctrl+C para salir.')
    print('En cualquier campo puedes presionar Enter para usar un valor aleatorio.\n')

    while True:
        try:
            rnd = random_patient()

            gender            = ask('Genero',
                                    options=['Male', 'Female'],
                                    default=rnd['gender'])
            age               = ask('Edad',
                                    cast=float,
                                    default=rnd['age'])
            hypertension      = ask('Hipertension',
                                    options=['0', '1'],
                                    cast=int,
                                    default=rnd['hypertension'])
            heart_disease     = ask('Enfermedad cardiaca',
                                    options=['0', '1'],
                                    cast=int,
                                    default=rnd['heart_disease'])
            ever_married      = ask('Casado alguna vez',
                                    options=['Yes', 'No'],
                                    default=rnd['ever_married'])
            work_type         = ask('Tipo de trabajo',
                                    options=['Govt_job', 'Never_worked', 'Private',
                                             'Self-employed', 'children'],
                                    default=rnd['work_type'])
            residence_type    = ask('Tipo de residencia',
                                    options=['Rural', 'Urban'],
                                    default=rnd['residence_type'])
            avg_glucose_level = ask('Glucosa promedio',
                                    cast=float,
                                    default=rnd['avg_glucose_level'])
            bmi               = ask('IMC',
                                    cast=float,
                                    default=rnd['bmi'])
            smoking_status    = ask('Tabaquismo',
                                    options=['Unknown', 'formerly smoked',
                                             'never smoked', 'smokes'],
                                    default=rnd['smoking_status'])

            row = build_row(age, hypertension, heart_disease, ever_married,
                            work_type, residence_type, avg_glucose_level,
                            bmi, smoking_status, gender)

            results = predict(models, row)
            print()
            print_results({'age': age,
                           'avg_glucose_level': avg_glucose_level,
                           'bmi': bmi}, results)
            print()

        except KeyboardInterrupt:
            print('\nSaliendo.')
            break
        except Exception as e:
            print(f'Error: {e}\n')


def main():
    parser = argparse.ArgumentParser(
        description='Prediccion de riesgo de stroke usando modelos entrenados.'
    )
    parser.add_argument(
        '--models', required=True,
        help='Directorio con los modelos guardados (lr.pkl, rf.pkl, *.keras)'
    )
    parser.add_argument('--age',               type=float)
    parser.add_argument('--gender',            default='Male')
    parser.add_argument('--hypertension',      type=int, choices=[0, 1])
    parser.add_argument('--heart_disease',     type=int, choices=[0, 1])
    parser.add_argument('--ever_married',      choices=['Yes', 'No'])
    parser.add_argument('--work_type',
                        choices=['Govt_job', 'Never_worked', 'Private',
                                 'Self-employed', 'children'])
    parser.add_argument('--residence_type',    choices=['Rural', 'Urban'])
    parser.add_argument('--avg_glucose_level', type=float)
    parser.add_argument('--bmi',               type=float)
    parser.add_argument('--smoking_status',
                        choices=['Unknown', 'formerly smoked',
                                 'never smoked', 'smokes'])
    parser.add_argument('--interactive', action='store_true',
                        help='Modo interactivo por consola')
    parser.add_argument('--random', action='store_true',
                        help='Genera un paciente completamente aleatorio y predice')

    args = parser.parse_args()

    print('Cargando modelos...')
    models = load_models(args.models)
    loaded = [k for k in models if k != 'scaler']
    print(f'Modelos cargados: {", ".join(loaded)}\n')

    if args.interactive:
        interactive_mode(models)
        return

    if args.random:
        p = random_patient()
        print('Paciente generado aleatoriamente:')
        for k, v in p.items():
            print(f'  {k}: {v}')
        print()
        row = build_row(p['age'], p['hypertension'], p['heart_disease'],
                        p['ever_married'], p['work_type'], p['residence_type'],
                        p['avg_glucose_level'], p['bmi'],
                        p['smoking_status'], p['gender'])
        results = predict(models, row)
        print_results({'age': p['age'],
                       'avg_glucose_level': p['avg_glucose_level'],
                       'bmi': p['bmi']}, results)
        return

    required = ['age', 'hypertension', 'heart_disease', 'ever_married',
                'work_type', 'residence_type', 'avg_glucose_level',
                'bmi', 'smoking_status']
    missing = [r for r in required if getattr(args, r) is None]
    if missing:
        print(f'Faltan argumentos: {", ".join(missing)}')
        print('Usa --interactive para el modo por consola.')
        sys.exit(1)

    row = build_row(
        args.age, args.hypertension, args.heart_disease, args.ever_married,
        args.work_type, args.residence_type, args.avg_glucose_level,
        args.bmi, args.smoking_status, args.gender
    )
    results = predict(models, row)
    print_results({'age': args.age,
                   'avg_glucose_level': args.avg_glucose_level,
                   'bmi': args.bmi}, results)


if __name__ == '__main__':
    main()