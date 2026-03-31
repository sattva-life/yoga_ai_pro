---
title: Sattvalife Yoga
emoji: 🧘
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
---

# Sattvalife Yoga
Django ML yoga pose detection app.

## 📁 Project Structure

```
sattvalife_yoga/
│
├── Guest/               # Guest app
├── User/                # User app
├── sattvalife_yoga/     # Main project settings
├── static/              # Static files (CSS, JS, Images)
├── Templates/           # HTML templates
├── db.sqlite3           # SQLite database
├── manage.py            # Django management script
└── req.txt              # Project dependencies
```

---

## ⚙️ Prerequisites

Make sure you have the following installed:

* Python (>= 3.8)
* pip (Python package manager)
* virtualenv (recommended)

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd sattvalife_yoga
```

### 2. Create and activate virtual environment

#### On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r req.txt
```

---

### 4. Apply migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

---

### 5. Create superuser (optional but recommended)

```bash
python manage.py createsuperuser
```

Follow the prompts to create admin credentials.

---

### 6. Run the development server

```bash
python manage.py runserver
```

---

## 🌐 Access the Application

* Main site: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* Admin panel: [http://127.0.0.1:8000/admin/](http://127.0.0.1:8000/admin/)

---

## 📦 Apps Included

* **Guest** – Handles guest user functionality
* **User** – Handles registered user features

---

## 🛠️ Static & Templates Setup

Ensure your `settings.py` includes:

```python
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / "static"]

TEMPLATES = [
    {
        'DIRS': [BASE_DIR / 'Templates'],
    },
]
```

---

## 🧪 Useful Commands

```bash
# Run server
python manage.py runserver

# Create migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
```

---

## 📝 Notes

* Default database: SQLite (`db.sqlite3`)
* Debug mode is enabled by default (disable in production)
* Update `ALLOWED_HOSTS` before deployment

---

## 📌 Future Improvements

* Add authentication system enhancements
* Improve UI/UX
* Deploy to cloud (AWS, Heroku, etc.)

---
"# yoga_ai" 
"# yoga_ai" 
"# yoga_ai" 
"# yoga_ai_pro" 
