import streamlit as st
import os

from outils import *

def main():
    st.set_page_config(page_title="PDF Summarizer")

    st.title("PDF Summarizing app ")
    st.write("Summarize your PDF file in a few seconds")
    st.divider()

    pdf = st.file_uploader("Upload your pdf file",type='pdf')

    submit = st.button("Generate Summazy")

    if submit:
        # response = summarizer(pdf)
        with st.spinner("🔎 Processing and summarizing..."):
                try:
                    response = summarizer(pdf)
                    st.success("✅ Summary generated successfully!")
                    st.subheader("📚 Summary of the file:")
                    st.write(response)
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
        # st.subheader('Summer of the file:')
        # st.write(response)

if __name__ == '__main__' :
    main()
