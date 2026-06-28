import streamlit as st

def main():
    keeptrack_day6 = st.Page("pages/day6.py", title = "Keeptrack day6")
    keeptrack_day7 = st.Page("pages/day7.py", title = "Keeptrack day7")
    pg = st.navigation([keeptrack_day6, keeptrack_day7])
    pg.run()
if __name__ == '__main__':
    main()