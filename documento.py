import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
np.random.seed(1234)
st.title("muebles")
datos=np.random.normal(0,1,size=(100,4))
data=pd.DataFrame(datos,columns=list("ABCD"))
st.dataframe(data)
e=np.random.normal(0,1,size=100)
y= data["A"]*2+data["B"]*3+data["C"]*4+data["D"]*0.3+10+e
x=data
model=DecisionTreeRegressor(max_depth=4)
model.fit(x,y)
st.subheader("A")
Val_a=st.slider("ingrese el valor de A",data["A"].min(),data["A"].max())
st.subheader("B")
Val_b=st.slider("ingrese el valor de B",data["B"].min(),data["B"].max())
st.subheader("C")
Val_c=st.slider("ingrese el valor de C",data["C"].min(),data["C"].max())
st.subheader("D")
Val_d=st.slider("ingrese el valor de D",data["D"].min(),data["D"].max())
valores=np.array([[Val_a,Val_b,Val_c,Val_d]])
p=model.predict([[Val_a,Val_b,Val_c,Val_d]])
st.write(p)