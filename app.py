# app.py
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# OpenAI (Responses API)
# pip install openai
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# =========================
# Page config
# =========================
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")


# =========================
# Utilities
# =========================
def _today_str() -> str:
    # Asia/Seoul 기준을 완벽히 맞추려면 zoneinfo를 쓰면 되지만,
    # Streamlit Cloud 환경에서도 무난하게 "오늘" 개념으로 쓰기 위해 local date 사용한다.
    return dt.date.today().isoformat()


def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _breed_from_dog_ceo_image_url(image_url: str) -> Optional[str]:
    """
    Dog CEO 이미지 URL에서 품종 추출.
    예: https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg
        -> hound afghan
    """
    try:
        if "/breeds/" not in image_url:
            return None
        part = image_url.split("/breeds/")[1].split("/")[0]  # hound-afghan
        part = part.replace("-", " ").strip()
        return part or None
    except Exception:
        return None


def get_dog_image() -> Optional[Dict]:
    """
    Dog CEO 랜덤 강아지 이미지 URL + 품종 가져오기.
    - 실패 시 None 반환
    - timeout=10
    """
    url = "https://dog.ceo/api/breeds/image/random"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "success":
            return None
        image_url = data.get("message")
        if not image_url:
            return None
        breed = _breed_from_dog_ceo_image_url(image_url)
        return {"image_url": image_url, "breed": breed}
    except Exception:
        return None


# =========================
# AI report
# =========================
COACH_SYSTEM_PROMPTS = {
    "스파르타 코치": """너는 엄격한 습관 코치다.
- 돌려 말하지 말고 핵심만 말한다.
- 핑계는 받아주지 않는다.
- 대신 실행 가능한 지시를 짧게 준다.
- 공격적 비난은 금지, 엄격하지만 건설적으로 말한다.""",
    "따뜻한 멘토": """너는 따뜻하고 현실적인 멘토다.
- 사용자를 다그치지 않는다.
- 작은 성취도 인정해준다.
- 부담을 줄이는 방식으로 다음 행동을 제안한다.
- 과한 감정 과잉은 하지 말고 담백하게 따뜻하게 말한다.""",
    "게임 마스터": """너는 RPG 게임 마스터다.
- 오늘을 '퀘스트 진행'으로 비유한다.
- 보상/레벨업/경험치 같은 표현을 섞어준다.
- 오글거리지 않게, 짧고 재밌게 진행한다.
- 사용자를 조롱하거나 과한 역할극은 금지다.""",
}

REPORT_FORMAT_INSTRUCTION = """아래 형식 그대로 한국어로 출력해라. (제목/라벨 고정)

[컨디션 등급] S/A/B/C/D 중 하나

[습관 분석]
- 잘한 점: (2~4줄)
- 아쉬운 점: (2~4줄)
- 한 가지 핵심 병목: (1줄)

[내일 미션]
1) (구체적이고 작은 행동)
2) (구체적이고 작은 행동)
3) (선택 미션 1개)

[오늘의 한마디]
- (짧게 1줄)
"""


def generate_report(
    openai_api_key: str,
    coach_style: str,
    mood: int,
    habits: Dict[str, bool],
    dog: Optional[Dict],
) -> Optional[str]:
    """
    습관+기분+강아지 품종을 모아서 OpenAI에 전달해 리포트 생성.
    - 실패 시 None
    - 모델: gpt-5-mini
    """
    if not openai_api_key:
        return None
    if OpenAI is None:
        return None

    system_prompt = COACH_SYSTEM_PROMPTS.get(coach_style, COACH_SYSTEM_PROMPTS["따뜻한 멘토"])

    # 입력 요약
    habit_lines = []
    for k, v in habits.items():
        habit_lines.append(f"- {k}: {'완료' if v else '미완료'}")
    habits_text = "\n".join(habit_lines)

    dog_text = "강아지 정보 없음"
    if dog:
        dog_text = f"강아지 품종(추정): {dog.get('breed') or '알 수 없음'}"

    user_prompt = f"""오늘 체크인 데이터다.

[기분] {mood}/10

[습관]
{habits_text}

[강아지]
{dog_text}

요청:
- 위 데이터만 바탕으로 코칭 리포트를 작성해라.
- 과장된 의학 조언 금지. 심리/건강 관련 단정 금지.
- 구체적인 행동 제안은 10분~30분 안에 가능한 수준으로.
- 아래 출력 형식을 반드시 지켜라.

{REPORT_FORMAT_INSTRUCTION}
"""

    try:
        client = OpenAI(api_key=openai_api_key)
        resp = client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (resp.output_text or "").strip()
        return text or None
    except Exception:
        return None


# =========================
# Session state init
# =========================
if "history" not in st.session_state:
    # 6일 데모 + 오늘(빈 값)으로 총 7일 만들기
    # 습관 5개 체크 개수 기반 달성률을 차트에 쓰기 좋게 만든다.
    today = dt.date.today()
    demo_days = [today - dt.timedelta(days=i) for i in range(6, 0, -1)]  # 6일 전 ~ 어제
    demo_moods = [6, 7, 5, 8, 6, 7]
    demo_done = [2, 3, 1, 4, 3, 2]  # 완료 습관 개수(0~5)

    rows = []
    for d, mood, done_cnt in zip(demo_days, demo_moods, demo_done):
        rows.append(
            {
                "date": d.isoformat(),
                "done_count": done_cnt,
                "achievement_pct": round(done_cnt / 5 * 100, 0),
                "mood": mood,
            }
        )
    # 오늘 데이터는 나중에 저장 버튼/생성 버튼 시 갱신
    rows.append(
        {
            "date": today.isoformat(),
            "done_count": 0,
            "achievement_pct": 0.0,
            "mood": 5,
        }
    )
    st.session_state.history = rows

if "last_report" not in st.session_state:
    st.session_state.last_report = None


if "last_dog" not in st.session_state:
    st.session_state.last_dog = None


# =========================
# Sidebar: API keys
# =========================
st.sidebar.header("🔑 API 키")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-...")

st.sidebar.caption("키는 로컬 세션에만 입력되고, 저장은 하지 않도록 구성했다.")


# =========================
# Main UI
# =========================
st.title("📊 AI 습관 트래커")
st.caption("오늘의 체크인을 남기고, AI 코치 리포트를 받아보면 된다.")

# --- 체크인 UI ---
st.subheader("✅ 오늘의 습관 체크인")

if "habit_names" not in st.session_state:
    st.session_state.habit_names = ["물 마시기", "공부/독서", "운동하기", "수면"]

habit_df = pd.DataFrame({"습관": st.session_state.habit_names})
edited_habits = st.data_editor(
    habit_df,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    column_config={"습관": st.column_config.TextColumn("지키고 싶은 습관", required=True)},
)
habit_names = [h.strip() for h in edited_habits["습관"].tolist() if isinstance(h, str) and h.strip()]
if not habit_names:
    habit_names = ["물 마시기"]
st.session_state.habit_names = habit_names

col_a, col_b = st.columns(2, gap="large")
habit_values: Dict[str, bool] = {}
for idx, habit_name in enumerate(habit_names):
    target_col = col_a if idx % 2 == 0 else col_b
    with target_col:
        habit_values[habit_name] = st.checkbox(habit_name, value=False)

mood = st.slider("🙂 오늘 기분 점수", min_value=1, max_value=10, value=7)

c1 = st.columns([1], gap="large")[0]
with c1:
    coach_style = st.radio("🎭 코치 스타일", ["스파르타 코치", "따뜻한 멘토", "게임 마스터"], horizontal=True)

done_count = sum(1 for v in habit_values.values() if v)
total_habits = max(1, len(habit_names))
achievement_pct = round(done_count / total_habits * 100, 0)


# --- Metrics ---
st.divider()
st.subheader("📈 오늘의 지표")

mcol1, mcol2, mcol3 = st.columns(3, gap="large")
with mcol1:
    st.metric("달성률", f"{int(achievement_pct)}%")
with mcol2:
    st.metric("달성 습관", f"{done_count}/{total_habits}")
with mcol3:
    st.metric("기분", f"{mood}/10")

# --- Save to session_state history (today row update) ---
def upsert_today(history_rows: List[Dict], done: int, pct: float, mood_value: int) -> List[Dict]:
    today = _today_str()
    updated = False
    new_rows = []
    for row in history_rows:
        if row.get("date") == today:
            new_rows.append(
                {
                    "date": today,
                    "done_count": done,
                    "achievement_pct": float(pct),
                    "mood": int(mood_value),
                }
            )
            updated = True
        else:
            new_rows.append(row)
    if not updated:
        new_rows.append(
            {
                "date": today,
                "done_count": done,
                "achievement_pct": float(pct),
                "mood": int(mood_value),
            }
        )
    # 날짜 정렬
    new_rows = sorted(new_rows, key=lambda r: r.get("date", ""))
    # 7일만 유지
    if len(new_rows) > 7:
        new_rows = new_rows[-7:]
    return new_rows


st.session_state.history = upsert_today(
    st.session_state.history,
    done=done_count,
    pct=achievement_pct,
    mood_value=mood,
)

# --- 7일 바 차트 ---
st.subheader("🗓️ 최근 7일 달성률")

df = pd.DataFrame(st.session_state.history)
if not df.empty and "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    chart_df = df.set_index("date")[["achievement_pct"]]
    st.bar_chart(chart_df)
else:
    st.info("아직 기록이 없다.")


# =========================
# Report generation
# =========================
st.divider()
st.subheader("🤖 AI 코치 리포트")

btn = st.button("컨디션 리포트 생성", type="primary", use_container_width=True)

if btn:
    with st.spinner("강아지/AI 리포트를 불러오는 중이다..."):
        dog = get_dog_image()
        report = generate_report(
            openai_api_key=openai_api_key,
            coach_style=coach_style,
            mood=mood,
            habits=habit_values,
            dog=dog,
        )
        st.session_state.last_dog = dog
        st.session_state.last_report = report

# --- Display results ---
dog = st.session_state.last_dog
report = st.session_state.last_report

rcol2 = st.columns(1, gap="large")[0]
with rcol2:
    st.markdown("### 🐶 오늘의 강아지")
    if dog and dog.get("image_url"):
        caption = f"품종(추정): {dog.get('breed') or '알 수 없음'}"
        st.image(dog["image_url"], caption=caption, use_container_width=True)
    else:
        st.info("강아지 이미지를 가져오지 못했다. (Dog CEO API 응답 실패)")

st.markdown("### 📝 리포트")
if report:
    st.markdown(report)
else:
    st.warning("리포트가 아직 없다. 버튼을 눌러 생성해봐라. (OpenAI 키/라이브러리 설치 여부 확인)")

# --- Share text ---
st.markdown("### 📎 공유용 텍스트")
share_lines = [
    f"📊 AI 습관 트래커 ({_today_str()})",
    f"코치: {coach_style}",
    f"달성률: {int(achievement_pct)}% ({done_count}/{total_habits}) / 기분: {mood}/10",
    "습관 체크:",
]
for name in habit_names:
    share_lines.append(f"- {name}: {'✅' if habit_values.get(name) else '⬜'}")
if dog:
    share_lines.append(f"강아지(추정 품종): {dog.get('breed') or '알 수 없음'}")

if report:
    share_lines.append("\n--- AI 코치 한줄 요약 ---")
    # 리포트에서 오늘의 한마디 라인을 최대한 뽑아보기 (없으면 첫 1~2줄)
    try:
        lines = [l.strip() for l in report.splitlines() if l.strip()]
        one = next((l for l in lines if l.startswith("-") and "한마디" not in l), None)
        share_lines.append(one if one else lines[0])
    except Exception:
        pass

st.code("\n".join(share_lines), language="text")


# =========================
# API 안내
# =========================
with st.expander("🔎 API 안내"):
    st.markdown(
        """
- **OpenAI API Key**: AI 코치 리포트 생성에 필요하다. (모델: `gpt-5-mini`)
- **Dog CEO API**: 랜덤 강아지 사진을 가져온다. (키 필요 없음)

문제 생기면 아래를 먼저 확인하면 된다.
1) 키를 제대로 붙여넣었는지 (공백/줄바꿈 포함 여부)
2) `pip install openai requests pandas` 설치가 되었는지
3) 네트워크 상태가 정상인지
"""
    )


# =========================
# Calendar + stamp
# =========================
st.divider()
st.subheader("📅 월간 달성 캘린더")

today = dt.date.today()
first_day = today.replace(day=1)
next_month = (first_day.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
last_day = next_month - dt.timedelta(days=1)

history_map = {row["date"]: row for row in st.session_state.history}

st.markdown(
    """
    <style>
    .calendar-grid {display:grid;grid-template-columns:repeat(7,1fr);gap:8px;}
    .calendar-cell {min-height:64px;border:1px solid #e5e7eb;border-radius:10px;padding:6px;background:#fff;position:relative;}
    .calendar-cell.empty {background:#f9fafb;border-style:dashed;}
    .day-num {font-size:12px;font-weight:600;color:#374151;}
    .stamp {position:absolute;right:6px;bottom:6px;font-size:20px;opacity:0.92;}
    .weekday {font-size:12px;text-align:center;color:#6b7280;font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)

week_labels = ["월", "화", "수", "목", "금", "토", "일"]
header_html = "".join([f"<div class='weekday'>{d}</div>" for d in week_labels])

cells_html = []
for _ in range(first_day.weekday()):
    cells_html.append("<div class='calendar-cell empty'></div>")

for day in range(1, last_day.day + 1):
    d = first_day.replace(day=day)
    key = d.isoformat()
    row = history_map.get(key)
    stamp = ""
    if row and safe_int(row.get("achievement_pct", 0), 0) == 100:
        stamp = "<div class='stamp'>💯</div>"
    cells_html.append(f"<div class='calendar-cell'><div class='day-num'>{day}</div>{stamp}</div>")

st.markdown(f"<div class='calendar-grid'>{header_html}{''.join(cells_html)}</div>", unsafe_allow_html=True)
