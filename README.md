# **Fast Tamping AI**

Fast Tamping AI is an intelligent solution designed for the global railway maintenance and tamping industry. Its objective is to improve decision-making, automation, and rail condition monitoring through modern artificial intelligence.

The system is powered by lightweight and efficient AI models that enable real-time analysis directly on-site, taking edge computing to a more advanced and practical level.

## **Functionality**
Fast Tamping AI the system performs rail lift analysis using a Recurrent Neural Network trained on a custom dataset. This model measures lift differences between rails, verifies whether the lift is within acceptable parameters, and enhances low-quality sensor signals to produce clearer and more reliable data. Using information from sixteen input channels and basic physical principles, such as acceleration vectors, the system estimates the lift of the left and right rails along the track.

In addition, Fast Tamping AI uses a multi-layer perceptron capable of processing eight input variables and producing two output values with millimeter-scale resolution. The network is activated using the ReLU function to ensure efficient and stable learning, and it is able to predict the required mechanical adjustment based on nonlinear input variables.

And finally the last model, fast Tamping AI incorporates a computer vision model capable of detecting the correct position of the steppers to ensure that tamping tines are inserted accurately into the ballast during the compaction process. This assists the tamping operation by improving precision and reducing the risk of misalignment.

 
## **Technology Stack**

Fast Tamping AI is developed using Python, Django, React Native, and PyTorch.

## **Summary**

By combining computer vision, signal processing, and machine learning, Fast Tamping AI improves tamping accuracy, reduces human error, increases maintenance efficiency, and contributes to smarter and safer railway infrastructure.

## **Install and run**
To install and run the model, it is necessary to follow the steps below:
1. Clone the repository
First, clone the repository and navigate to the project folder. You will need to initialize two different environments and install some dependencies.
git clone <(https://github.com/Yizuz02/SodaPop_Plasser/edit/main)>
cd <repository_folder>
2. Backend setup (Django)
Go to the backend directory, create a virtual environment, install Django and dependencies, create a superuser, and start the server:
cd backend
python3 -m venv backend-env
source backend-env/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
3. Frontend setup (React Native)
Install Node.js using NVM and initialize the React Native frontend:
cd ../frontend
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install node
npm install
npm start
4. Model setup (PyTorch and OpenCV)
Go to the model directory, create a virtual environment, and install Torch, PyTorch, and OpenCV:
cd ../model
python3 -m venv model-env
source model-env/bin/activate
pip install torch torchvision torchaudio
pip install opencv-python
5. Install additional Python libraries
Finally, make sure that libraries such as pandas, numpy, scikit-learn, and others used in the project are installed:
pip install numpy pandas scikit-learn matplotlib
