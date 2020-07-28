''' ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'salary', 'IT', 'accounting', 'hr',
       'management', 'marketing', 'product_mng', 'sales', 'support',
       'technical'] '''



import pickle
import streamlit as st
import numpy as np

knn=pickle.load(open('knn.pkl','rb'))
log_reg=pickle.load(open('hr_log_reg.pkl','rb'))
dt=pickle.load(open('dt.pkl','rb'))
svm=pickle.load(open('svm.pkl','rb'))

def main():
    IT=accounting=hr=management=marketing=sales=product_mng=support=technical=0
    st.title("HR Analysis App")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Employee churn predictor</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    activities=["K Nearest Neighbors","Support Vector Machines","Decision Tree"]
    options=st.sidebar.selectbox('Which model would you like to use',activities)
    st.subheader(options)
    satisfaction_level=st.text_input("Satisfaction level")
    last_evaluation=st.text_input("last evaluation")
    number_project=st.text_input("number project")
    average_monthly_hours=st.text_input("average monthly hours")
    time_spend_company=st.text_input("time spent at company")
    work_accident=st.text_input("work accident")
    promotion_last_5years=st.text_input("Promotion in last "+ str(time_spend_company) +" years")
    salary_type=["low","medium","high"]
    sal_type=st.selectbox("Select employee's salary type",salary_type)
    if sal_type=="low":
        salary=1
    elif sal_type=="medium":
        salary=2
    else:
        salary=3   
    #IT=st.text_input("IT","Type here")
    #accounting=st.text_input("accounting","Type here")
    #hr=st.text_input("hr","Type here")
    #management=st.text_input("management","Type here")
    #marketing=st.text_input("marketing","Type here")
    #product_mng=st.text_input("product management","Type here")
    #sales=st.text_input("Sales","Type here")
    #support=st.text_input("Support","Type here")
    #technical=st.text_input("Technical","Type here")
    departments=['IT', 'accounting', 'hr',
       'management', 'marketing', 'product_mng', 'sales', 'support',
       'technical']
    dept=st.selectbox("Select employee's department",departments)
    if dept=="IT":
        IT=1
    elif dept=="accounting":
        accounting=1
    elif dept=="hr":
        hr=1
    elif dept=="management":
        management=1
    elif dept=="marketing":
        marketing=1
    elif dept=="product_mng":
        product_mng=1
    elif dept=="sales":
        sales=1
    elif dept=="support":
        support=1
    elif dept=="technical":
        technical=1  
          
    left_html="""
    <div style="background-color:#F08080;padding:10px">
    <h2 style="color:black;text-align:center;"> The employee is likely to leave</h2>
    </div>
    """
    stay_html="""
    <div style="background-color:#F4D03F;padding:10px">
    <h2 style="color:black;text-align:center;"> The employee is going to stay</h2>
    </div>
    """
    inputs=np.array([[satisfaction_level, last_evaluation, number_project,
       average_monthly_hours, time_spend_company, work_accident,
       promotion_last_5years, salary, IT, accounting, hr,
       management, marketing, product_mng, sales, support,
       technical]])
    if st.button('classify'):
        if options=='K Nearest Neighbors':
            result=knn.predict(inputs)
            st.success('The output is {}'.format(result))
            if result==0:
                st.markdown(stay_html,unsafe_allow_html=True)
            else:
                st.markdown(left_html,unsafe_allow_html=True)
        elif options=='Support Vector Machines':
            result=svm.predict(inputs)
            st.success('The output is {}'.format(result))
            if result==0:
                st.markdown(stay_html,unsafe_allow_html=True)
            else:
                st.markdown(left_html,unsafe_allow_html=True)
        else:
            result=dt.predict(inputs)
            st.success('The output is {}'.format(result))
            if result==0:
                st.markdown(stay_html,unsafe_allow_html=True)
            else:
                st.markdown(left_html,unsafe_allow_html=True)
            
    if st.button("About"):
        st.text("Lets Learn streamlit")
        st.text("this app was built with Streamlit")

if __name__=='__main__':
    main()
        
