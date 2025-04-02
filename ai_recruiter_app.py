# -----------------------------
# Dropbox Setup
# -----------------------------
if "DROPBOX_TOKEN" not in st.session_state:
    st.session_state["DROPBOX_TOKEN"] = ""

st.session_state["DROPBOX_TOKEN"] = st.text_input(
    "Enter your Dropbox token",
    type="password",
    value=st.session_state["DROPBOX_TOKEN"]
)

DROPBOX_TOKEN = st.session_state["DROPBOX_TOKEN"]
dbx = None
selected_member_id = None

# Connection Test
if DROPBOX_TOKEN:
    try:
        dbx = dropbox.Dropbox(DROPBOX_TOKEN)
        current_account = dbx.users_get_current_account()
        st.sidebar.success(f"✅ Connected to Dropbox as {current_account.name.display_name} ({current_account.email})")
    except dropbox.exceptions.BadInputError as e:
        if "Dropbox-API-Select-User" in str(e):
            try:
                dbx_team = dropbox.DropboxTeam(DROPBOX_TOKEN)
                members = dbx_team.team_members_list().members

                member_options = {}
                for m in members:
                    status_tag_func = getattr(m.profile.status, 'tag', None)
                    if callable(status_tag_func) and status_tag_func() == "active":
                        member_options[m.profile.email] = m.profile.team_member_id
                    elif hasattr(m.profile.status, 'is_active') and m.profile.status.is_active:
                        member_options[m.profile.email] = m.profile.team_member_id

                if member_options:
                    selected_email = st.sidebar.selectbox("👤 Select a team member to act as", list(member_options.keys()))
                    selected_member_id = member_options[selected_email]

                    dbx = dbx_team.as_user(selected_member_id)
                    current_account = dbx.users_get_current_account()
                    st.sidebar.success(f"✅ Acting as: {current_account.name.display_name} ({current_account.email})")
                else:
                    st.sidebar.error("❌ No active team members found.")

            except Exception as member_error:
                st.sidebar.error(f"❌ Failed to impersonate team member: {member_error}")
                st.session_state["DROPBOX_TOKEN"] = ""
                dbx = None
        else:
            st.sidebar.error(f"❌ Failed to connect to Dropbox: {e}")
            st.session_state["DROPBOX_TOKEN"] = ""
            dbx = None
    except Exception as general_error:
        st.sidebar.error(f"❌ General Dropbox connection error: {general_error}")
        dbx = None

    if dbx:
        try:
            current_account = dbx.users_get_current_account()
            st.sidebar.success(f"✅ Connected to Dropbox as {current_account.name.display_name}")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to fetch Dropbox account details: {e}")
            dbx = None
else:
    st.sidebar.error("❌ No Dropbox token entered. Please paste your token above.")
