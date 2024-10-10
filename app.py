
from flask import Flask, redirect, request, render_template, jsonify, session, url_for
import random
import csv
import io
from datetime import datetime
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

 
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = './uploads' 
# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#Form 
#DATA_FILE = 'attendees.csv'

# Ensure the CSV file has a header row
''''
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Role', 'Name', 'Email', 'Skills', 'Background', 'Domain', 'Interests'])
'''


# Memory store for generated groups and logs
generated_groups_log = []
logs = []

# Helper function to create a unique identifier for each group
def create_group_key(groups):
    return tuple(sorted(tuple(sorted(group)) for group in groups))

def generate_balanced_unique_groups(professionals, students, group_size, max_students_per_group):
    groups = []
    num_groups = (len(professionals) + len(students)) // group_size

    max_attempts = 10
    attempt = 0

    while attempt < max_attempts:
        random.shuffle(professionals)
        random.shuffle(students)

        # Step 1: Assign exactly one student to each group to ensure every group has at least one student.
        temp_groups = [[students.pop()] for _ in range(min(num_groups, len(students)))]

        # Step 2: Distribute remaining students to groups randomly while respecting max_students_per_group.
        while students:
            # Pick a random group that still has room for more students
            available_groups = [group for group in temp_groups if len(group) < max_students_per_group]
            if available_groups:
                random.choice(available_groups).append(students.pop())
            else:
                break  # No more available groups with room for students

        # Step 3: Fill remaining spots in the groups with professionals.
        for group in temp_groups:
            while len(group) < group_size and professionals:
                group.append(professionals.pop())

        # Step 4: Handle remaining professionals or students, if any.
        remaining_members = students + professionals
        index = 0
        while remaining_members:
            temp_groups[index % len(temp_groups)].append(remaining_members.pop())
            index += 1

        group_key = create_group_key(temp_groups)

        if group_key not in generated_groups_log:
            generated_groups_log.append(group_key)
            log_group_change(temp_groups, group_size)
            return temp_groups

        attempt += 1

    return []



# Log each group change
def log_group_change(groups, change_interval):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for index, group in enumerate(groups, 1):
        log_entry = {
            'time': current_time,
            'group_number': f'Group {index}',
            'members': ', '.join(group),
            'change_interval': change_interval
        }
        logs.append(log_entry)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route to display the registration form
@app.route('/matching-groups', methods=['GET'])
def display_matching_groups():
    # Load and display matching groups logic
    return render_template('matching-groups.html')



@app.route('/create_groups', methods=['POST'])
def create_groups():
    data = request.get_json()
    group_size = int(data.get('group_size'))
    max_students_per_group = int(data.get('max_students_per_group'))
    professionals = data.get('professionals')
    students = data.get('students')
    change_interval = data.get('change_interval')

    groups = generate_balanced_unique_groups(professionals, students, group_size, max_students_per_group)

    return jsonify(groups)

@app.route('/download_log', methods=['GET'])
def download_log():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['time', 'group_number', 'members', 'change_interval'])
    writer.writeheader()
    writer.writerows(logs)

    response = app.response_class(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=group_log.csv'}
    )
    return response

@app.route('/clear_log', methods=['POST'])
def clear_log():
    global logs
    logs = []
    return jsonify({'status': 'Log cleared'})

@app.route('/get_log', methods=['GET'])
def get_log():
    return jsonify(logs)

@app.route('/test', methods=['GET'])
def test():
    return 'Test route is working!'



# Step 1: Extract Features with Domain and Skills Weights
def refine_feature_matrix(df, domain_weight=3, skill_weight=1):
  
    enc = OneHotEncoder(sparse_output=False)
    tfidf_vectorizer = TfidfVectorizer()

    domain_encoded = enc.fit_transform(df[['Domain']]) * domain_weight
    skills_encoded = tfidf_vectorizer.fit_transform(df['Skills'].apply(lambda x: x.replace(',', ' '))) * skill_weight

    features = np.hstack([
        skills_encoded.toarray(),
        domain_encoded
    ])

    return features

# Step 2: Clustering Algorithm
def cluster_by_similarity(features, num_clusters=10):

    clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='cosine', linkage='average')
    cluster_labels = clustering.fit_predict(features)
    return cluster_labels

# Step 3: Balance Groups by Roles
def balance_groups(df, cluster_labels, min_group_size=3, max_group_size=5):
    df['Cluster'] = cluster_labels
    students = df[df['Role'].str.lower() == 'student'].index.tolist()
    professionals = df[df['Role'].str.lower() == 'professional'].index.tolist()
    
    np.random.shuffle(students)
    np.random.shuffle(professionals)

    balanced_groups = []
    ungrouped = students + professionals

    for cluster in np.unique(cluster_labels):
        group = []
        cluster_members = df[df['Cluster'] == cluster].index.tolist()

        for member in cluster_members:
            if len(group) < max_group_size:
                if df.loc[member, 'Role'].lower() == 'student' and len([x for x in group if df.loc[x, 'Role'].lower() == 'student']) < 2:
                    group.append(member)
                elif df.loc[member, 'Role'].lower() == 'professional':
                    group.append(member)
        
        if len(group) >= min_group_size:
            balanced_groups.append(group)
            ungrouped = [x for x in ungrouped if x not in group]

    balanced_groups = assign_ungrouped_to_closest(df, ungrouped, balanced_groups, max_group_size)

    return balanced_groups

# Step 4: Assign Remaining Profiles to Closest Group
def assign_ungrouped_to_closest(df, ungrouped, balanced_groups, max_group_size):
    for member in ungrouped:
        best_group = None
        best_similarity = -1
        
        for group in balanced_groups:
            if len(group) < max_group_size:
                member_features = df.loc[member, 'Features'].reshape(1, -1)
                group_features = np.mean(df.loc[group, 'Features'].values.tolist(), axis=0).reshape(1, -1)
                similarity = cosine_similarity(member_features, group_features)[0][0]

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_group = group

        if best_group is not None:
            best_group.append(member)
        else:
            balanced_groups.append([member])

    return balanced_groups

# Step 5: Assign Group Labels to DataFrame
def assign_groups_to_df(df, balanced_groups):
    group_labels = [None] * len(df)
    for i, group in enumerate(balanced_groups):
        for member in group:
            group_labels[member] = i
    df['Group'] = group_labels

# Step 6: Flask Routes
@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    if 'dataset' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['dataset']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    return redirect(url_for('process_dataset', filename=file.filename))

@app.route('/process-dataset/<filename>', methods=['GET'])
def process_dataset(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path)

    features = refine_feature_matrix(df)
    df['Features'] = list(features)

    cluster_labels = cluster_by_similarity(features, num_clusters=5)
    df['Cluster'] = cluster_labels

    balanced_groups = balance_groups(df, cluster_labels)
    assign_groups_to_df(df, balanced_groups)

    # Prepare groups for display
    groups = df.groupby('Group')[['Name', 'Skills']].apply(lambda x: x.to_dict(orient='records')).to_dict()

    return render_template('display_groups.html', groups=groups)





@app.route('/matching-groups', methods=['GET'])
def matching_groups():
    return render_template('matching-groups.html')

@app.route('/display_groups', methods=['GET'])
def display_groups():
    return render_template('display_groups.html')


if __name__ == "__main__":
    app.run(debug=True)
