from typing import Annotated
from langchain_core.tools import tool
from typing import Dict, Any, Union
from pykrx.stock.stock_api import get_market_ohlcv, get_nearest_business_day_in_a_week, get_market_cap, \
    get_market_fundamental_by_date, get_market_trading_volume_by_date
from pykrx.website.krx.market.wrap import get_market_ticker_and_name
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm

TICKER_MAP: Dict[str, str] = {}

@tool()
def load_all_tickers() -> Dict[str, str]:
    """Loads all ticker symbols and names for KOSPI and KOSDAQ into memory.

    Returns:
        Dict[str, str]: A dictionary mapping tickers to stock names.
        Example: {"005930": "삼성전자", "035720": "카카오", ...}
    """
    try:
        global TICKER_MAP

        # If TICKER_MAP already has data, return it
        if TICKER_MAP:
            logging.debug(f"Returning cached ticker information with {len(TICKER_MAP)} stocks")
            return TICKER_MAP

        logging.debug("No cached data found. Loading KOSPI/KOSDAQ ticker symbols")

        # Retrieve data based on today's date
        today = get_nearest_business_day_in_a_week()
        logging.debug(f"Reference date: {today}")

        # get_market_ticker_and_name() returns a Series,
        # where the index is the ticker and the values are the stock names
        kospi_series = get_market_ticker_and_name(today, market="KOSPI")
        kosdaq_series = get_market_ticker_and_name(today, market="KOSDAQ")

        # Convert Series to dictionaries and merge them
        TICKER_MAP.update(kospi_series.to_dict())
        TICKER_MAP.update(kosdaq_series.to_dict())

        logging.debug(f"Successfully stored information for {len(TICKER_MAP)} stocks")
        return TICKER_MAP

    except Exception as e:
        error_message = f"Failed to retrieve ticker information: {str(e)}"
        logging.error(error_message)
        return {"error": error_message}

@tool
def get_stock_ohlcv(fromdate: Union[str, int], todate: Union[str, int], ticker: Union[str, int], adjusted: bool = True) -> Dict[str, Any]:
    """Retrieves OHLCV (Open/High/Low/Close/Volume) data for a specific stock.

    Args:
        fromdate (str): Start date for retrieval (YYYYMMDD)
        todate   (str): End date for retrieval (YYYYMMDD)
        ticker   (str): Stock ticker symbol
        adjusted (bool, optional): Whether to use adjusted prices (True: adjusted, False: unadjusted). Defaults to True.

    Returns:
        DataFrame:
            >> get_stock_ohlcv("20210118", "20210126", "005930")
                            Open     High     Low    Close   Volume
            Date
            2021-01-26  89500  94800  89500  93800  46415214
            2021-01-25  87300  89400  86800  88700  25577517
            2021-01-22  89000  89700  86800  86800  30861661
            2021-01-21  87500  88600  86500  88100  25318011
            2021-01-20  89000  89000  86500  87200  25211127
            2021-01-19  84500  88000  83600  87000  39895044
            2021-01-18  86600  87300  84100  85000  43227951
    """
    # Validate and convert date format
    def validate_date(date_str: Union[str, int]) -> str:
        try:
            if isinstance(date_str, int):
                date_str = str(date_str)
            # Convert if in YYYY-MM-DD format
            if '-' in date_str:
                parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
                return parsed_date.strftime('%Y%m%d')
            # Validate if in YYYYMMDD format
            datetime.strptime(date_str, '%Y%m%d')
            return date_str
        except ValueError:
            raise ValueError(f"Date must be in YYYYMMDD format. Input value: {date_str}")

    def validate_ticker(ticker_str: Union[str, int]) -> str:
        if isinstance(ticker_str, int):
            return str(ticker_str)
        return ticker_str

    try:
        fromdate = validate_date(fromdate)
        todate = validate_date(todate)
        ticker = validate_ticker(ticker)

        logging.debug(f"Retrieving stock OHLCV data: {ticker}, {fromdate}-{todate}, adjusted={adjusted}")

        # Call get_market_ohlcv (changed adj -> adjusted)
        df = get_market_ohlcv(fromdate, todate, ticker, adjusted=adjusted)

        # Convert DataFrame to dictionary
        result = df.to_dict(orient='index')

        # Convert datetime index to string and sort in reverse
        sorted_items = sorted(
            ((k.strftime('%Y-%m-%d'), v) for k, v in result.items()),
            reverse=True
        )
        result = dict(sorted_items)

        return result

    except Exception as e:
        error_message = f"Data retrieval failed: {str(e)}"
        logging.error(error_message)
        return {"error": error_message}
    
    
# @tool
# def search_company_ticker(company_name: Annotated[str, "회사명 (예: 삼성전자, 카카오)"]) -> str:
#     """회사명으로 종목코드 검색 - 기본 도구"""
#     basic_mapping = {
#         "삼성전자": "005930",
#         "SK하이닉스": "000660", 
#         "NAVER": "035420",
#         "네이버": "035420",
#         "카카오": "035720",
#         "LG에너지솔루션": "373220"
#     }
    
#     ticker = basic_mapping.get(company_name)
#     if ticker:
#         return f"{company_name}의 종목코드는 {ticker}입니다."
#     else:
#         return f"{company_name}의 종목코드를 찾을 수 없습니다. 정확한 회사명을 확인해주세요."

@tool
def get_stock_price_change(ticker: Union[str, int], days: int) -> Dict[str, Any]:
    """주가 변동률을 계산합니다.
    
    Args:
        ticker (str): 종목코드
        days (int): 분석할 기간 (일수)
    
    Returns:
        Dict[str, Any]: 변동률 분석 결과
        Example: {
            "ticker": "005930",
            "period": "30일",
            "start_price": 70000,
            "end_price": 75000,
            "change_amount": 5000,
            "change_rate": 7.14,
            "max_price": 78000,
            "min_price": 68000,
            "volatility": 12.5
        }
    """
    from datetime import datetime, timedelta
    
    def validate_ticker(ticker_str: Union[str, int]) -> str:
        if isinstance(ticker_str, int):
            return str(ticker_str)
        return ticker_str
    
    try:
        ticker = validate_ticker(ticker)
        
        # 현재 날짜에서 지정된 일수만큼 이전 날짜 계산
        # 주말/공휴일을 고려하여 더 넉넉한 기간으로 설정
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)  # 여유분 추가
        
        # 날짜를 YYYYMMDD 형식으로 변환
        fromdate = start_date.strftime('%Y%m%d')
        todate = end_date.strftime('%Y%m%d')
        
        logging.debug(f"Calculating price change for {ticker} from {fromdate} to {todate}")
        
        # OHLCV 데이터 조회 (직접 pykrx API 호출)
        df = get_market_ohlcv(fromdate, todate, ticker, adjusted=True)
        
        logging.debug(f"DataFrame shape: {df.shape}")
        logging.debug(f"DataFrame columns: {df.columns.tolist()}")
        logging.debug(f"DataFrame head:\n{df.head()}")
        
        # DataFrame을 dictionary로 변환
        ohlcv_data = df.to_dict(orient='index')
        
        logging.debug(f"Raw ohlcv_data keys: {list(ohlcv_data.keys())[:5]}")  # 처음 5개 키만
        
        # datetime index를 string으로 변환하고 역순 정렬
        sorted_items = sorted(
            ((k.strftime('%Y-%m-%d'), v) for k, v in ohlcv_data.items()),
            reverse=True
        )
        ohlcv_data = dict(sorted_items)
        
        logging.debug(f"Processed ohlcv_data keys: {list(ohlcv_data.keys())[:5]}")  # 처음 5개 키만
        
        if not ohlcv_data:
            return {"error": "No data available for the specified period"}
        
        # 날짜순으로 정렬 (오래된 것부터)
        sorted_dates = sorted(ohlcv_data.keys())
        
        if len(sorted_dates) < 2:
            return {"error": "Insufficient data for analysis"}
        
        # 요청된 기간에 맞는 데이터만 필터링 (최근 N일)
        # 최근 데이터부터 역순으로 정렬된 상태에서 요청된 일수만큼만 사용
        recent_dates = sorted_dates[-days:] if len(sorted_dates) >= days else sorted_dates
        
        if len(recent_dates) < 2:
            return {"error": "Insufficient recent data for analysis"}
        
        # 시작일과 종료일 데이터
        start_data = ohlcv_data[recent_dates[0]]
        end_data = ohlcv_data[recent_dates[-1]]
        
        logging.debug(f"Start data: {start_data}")
        logging.debug(f"End data: {end_data}")
        
        start_price = start_data.get('종가', 0)
        end_price = end_data.get('종가', 0)
        
        logging.debug(f"Start price: {start_price}, End price: {end_price}")
        
        if start_price == 0:
            return {
                "error": "Invalid start price data", 
                "debug_info": {
                    "start_date": recent_dates[0],
                    "end_date": recent_dates[-1],
                    "start_data": start_data,
                    "end_data": end_data,
                    "total_data_points": len(ohlcv_data),
                    "recent_data_points": len(recent_dates),
                    "requested_days": days
                }
            }
        
        # 변동률 계산
        change_amount = end_price - start_price
        change_rate = (change_amount / start_price) * 100
        
        # 최고가, 최저가 찾기
        max_price = max([data.get('고가', 0) for data in ohlcv_data.values()])
        min_price = min([data.get('저가', float('inf')) for data in ohlcv_data.values()])
        
        # 변동성 계산 (최고가-최저가)/시작가 * 100
        volatility = ((max_price - min_price) / start_price) * 100
        
        result = {
            "ticker": ticker,
            "period": f"{days}일",
            "start_date": recent_dates[0],
            "end_date": recent_dates[-1],
            "start_price": start_price,
            "end_price": end_price,
            "change_amount": change_amount,
            "change_rate": round(change_rate, 2),
            "max_price": max_price,
            "min_price": min_price,
            "volatility": round(volatility, 2),
            "data_points": len(recent_dates),
            "total_available_data": len(ohlcv_data)
        }
        
        return result
        
    except Exception as e:
        error_message = f"Price change calculation failed: {str(e)}"
        logging.error(error_message)
        return {"error": error_message}

@tool
def get_ai_stock_recommendation(ticker: Union[str, int], investment_style: str = "중립적") -> Dict[str, Any]:
    """AI 주식 분석가의 종합적인 투자 점수 분석을 제공합니다.
    
    **[중요]** 사용자가 'investment_style' 값을 명시하지 않았다면, 이 도구를 바로 호출하지 마세요.
    대신, 사용자에게 "어떤 투자 성향으로 분석해 드릴까요? (공격적, 중립적, 보수적)" 와 같이 반드시 먼저 질문해야 합니다.
    
    Args:
        ticker (str): 종목코드 (예: "005930", "000660")
        investment_style (str): 사용자의 투자 성향 ("공격적", "중립적", "보수적"). 이 값은 사용자 질문을 통해 받아야 합니다.
    
    Returns:
        Dict[str, Any]: 투자 점수 및 분석 결과
        Example: {
            "ticker": "005930",
            "investment_score": 0.515,
            "investment_grade": "B (보통)",
            "recommendation": "추천 - 적정한 투자 기회",
            "analysis": {
                "alpha": 7.17,
                "beta": 1.208,
                "sharpe_ratio": 0.335,
                "volatility": 34.36,
                "annual_return": 15.23,
                "market_return": 12.45
            },
            "reasons": [
                "양호한 알파 (7.2%) - 시장 대비 초과 수익",
                "양호한 샤프 비율 (0.34) - 위험 대비 적정",
                "적정 베타 (1.21) - 시장과 균형적"
            ]
        }
    """
    from datetime import timedelta
    
    # KOSPI 대표 종목들 (시장 지수 대신 사용)
    market_tickers = ["005930", "000660", "035420", "035720", "207940"]  # 삼성전자, SK하이닉스, 네이버, 카카오, 삼성바이오로직스
    risk_free_rate = 0.03
    days = 252
    
    def validate_ticker(ticker_str: Union[str, int]) -> str:
        if isinstance(ticker_str, int):
            return str(ticker_str)
        return ticker_str
    
    def validate_investment_style(style: str) -> str:
        """투자 성향 검증"""
        valid_styles = ["공격적", "중립적", "보수적"]
        if style not in valid_styles:
            return "중립적"  # 기본값
        return style
    
    try:
        ticker = validate_ticker(ticker)
        investment_style = validate_investment_style(investment_style)
        
        # 데이터 수집 기간 설정 (252 거래일을 정확히 얻기 위해 충분한 기간 설정)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)
        
        fromdate = start_date.strftime('%Y%m%d')
        todate = end_date.strftime('%Y%m%d')
        
        logging.debug(f"Fetching data for {ticker} from {fromdate} to {todate}")
        
        # 주식 데이터 수집
        stock_data = get_market_ohlcv(fromdate, todate, ticker, adjusted=True)
        
        if stock_data.empty:
            return {"error": f"데이터 수집 실패: {ticker} 데이터가 없습니다", "score": 0.0}
        
        if len(stock_data) >= days:
            stock_data = stock_data.tail(days)
            logging.debug(f"Using exactly {days} trading days for analysis")
        else:
            logging.warning(f"Only {len(stock_data)} trading days available, using all data")
        
        market_returns_list = []
        for market_ticker in market_tickers:
            try:
                market_data = get_market_ohlcv(fromdate, todate, market_ticker, adjusted=True)
                if not market_data.empty:
                    # 주식 데이터와 동일한 기간으로 맞춤
                    if len(market_data) >= days:
                        market_data = market_data.tail(days)
                    market_returns_list.append(market_data['종가'].pct_change().dropna())
            except:
                continue
        
        if not market_returns_list:
            return {"error": "시장 데이터 수집 실패", "score": 0.0}
        
        # 시장 수익률 계산 (여러 종목의 평균)
        market_returns = pd.concat(market_returns_list, axis=1).mean(axis=1)
        
        # 수익률 계산
        stock_returns = stock_data['종가'].pct_change().dropna()
        
        # 데이터 정렬 및 결합
        data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        data.columns = ['Stock', 'Market']
        
        if len(data) < 30:  # 최소 30일 데이터 필요
            return {"error": f"데이터 부족: 최소 30일 데이터가 필요하지만 {len(data)}일만 있습니다", "score": 0.0}
        
        # 기본 통계 계산
        stock_annual_return = stock_returns.mean() * 252
        market_annual_return = market_returns.mean() * 252
        stock_volatility = stock_returns.std() * np.sqrt(252)
        
        # 베타 및 알파 계산 (회귀분석)
        X = sm.add_constant(data['Market'])
        y = data['Stock']
        model = sm.OLS(y, X).fit()
        
        beta = model.params['Market']
        alpha = model.params['const'] * 252  # 연간화
        
        # 샤프 비율 계산
        stock_sharpe = (stock_annual_return - risk_free_rate) / stock_volatility
        
        # 투자 점수 계산 (0~1 사이)
        score = _calculate_investment_score_kr(
            alpha=alpha,
            beta=beta,
            sharpe=stock_sharpe,
            volatility=stock_volatility,
            style=investment_style
        )
        
        # 투자 등급 결정
        grade = _get_investment_grade_kr(score)
        
        # 투자 추천 이유
        reasons = _get_investment_reasons_kr(
            alpha=alpha,
            beta=beta,
            sharpe=stock_sharpe,
            volatility=stock_volatility,
        )
        
        return {
            "ticker": ticker,
            "investment_score": round(score, 3),
            "investment_grade": grade,
            "analysis": {
                "alpha": round(alpha * 100, 2),
                "beta": round(beta, 3),
                "sharpe_ratio": round(stock_sharpe, 3),
                "volatility": round(stock_volatility * 100, 2),
                "correlation": round(data.corr().iloc[0, 1], 3),
                "r_squared": round(model.rsquared, 3),
                "annual_return": round(stock_annual_return * 100, 2),
                "market_return": round(market_annual_return * 100, 2)
            },
            "reasons": reasons,
            "recommendation": _get_final_recommendation_kr(score, grade)
        }
        
    except Exception as e:
        error_message = f"분석 실패: {str(e)}"
        logging.error(error_message)
        return {"error": error_message, "score": 0.0}

def _calculate_investment_score_kr(alpha: float, beta: float, sharpe: float, 
                                 volatility: float, style: str) -> float:
    """0~1 사이의 투자 점수 계산 (한국 시장용)
    변동성 (Volatility): "이 주식 자체의 위험은 얼마나 큰가?" (절대적 위험)
    베타 (Beta): "시장이 움직일 때, 이 주식은 얼마나 민감하게 반응하는가?" (시장 위험)
    알파 (Alpha): "시장 위험을 감안했을 때, 시장보다 얼마나 더 잘했는가?" (초과 수익 능력)
    샤프 지수 (Sharpe Ratio): "총 위험 1단위당 얼마의 초과 수익을 얻었는가?" (위험 대비 수익 효율성)
    """
    score = 0.0
    weights = {
        # 성향: [알파, 샤프, 베타, 변동성] 가중치
        "공격적": [0.45, 0.35, 0.10, 0.10],
        "중립적": [0.35, 0.30, 0.20, 0.15],
        "보수적": [0.15, 0.25, 0.25, 0.35],
    }
    w_alpha, w_sharpe, w_beta, w_vol = weights[style]

    # 1. 알파 점수 (Alpha) - 높을수록 좋음
    if alpha > 0.1: alpha_score = 1.0
    elif alpha > 0.05: alpha_score = 0.8
    elif alpha > 0: alpha_score = 0.6
    else: alpha_score = 0.2

    # 2. 샤프 비율 점수 (Sharpe Ratio) - 높을수록 좋음
    if sharpe > 1.5: sharpe_score = 1.0
    elif sharpe > 1.0: sharpe_score = 0.8
    elif sharpe > 0.5: sharpe_score = 0.6
    else: sharpe_score = 0.2

    # 3. 베타 점수 (Beta) - 성향에 따라 기준 변경
    if style == "공격적":
        if beta > 1.3: beta_score = 1.0
        elif beta > 1.0: beta_score = 0.8
        elif beta > 0.8: beta_score = 0.5
        else: beta_score = 0.2
    elif style == "중립적":
        if 0.9 <= beta <= 1.1: beta_score = 1.0
        elif 0.8 <= beta <= 1.2: beta_score = 0.8
        elif 0.6 <= beta <= 1.5: beta_score = 0.5
        else: beta_score = 0.2
    else:  # 보수적
        if beta < 0.7: beta_score = 1.0
        elif beta < 0.9: beta_score = 0.8
        elif beta < 1.1: beta_score = 0.5
        else: beta_score = 0.2
        
    # 4. 변동성 점수 (Volatility) - 성향에 따라 기준 변경
    if style == "공격적":
        if volatility < 0.4: vol_score = 1.0
        elif volatility < 0.5: vol_score = 0.7
        else: vol_score = 0.3
    elif style == "중립적":
        if volatility < 0.25: vol_score = 1.0
        elif volatility < 0.35: vol_score = 0.8
        elif volatility < 0.45: vol_score = 0.5
        else: vol_score = 0.2
    else:  # 보수적
        if volatility < 0.20: vol_score = 1.0
        elif volatility < 0.30: vol_score = 0.8
        elif volatility < 0.40: vol_score = 0.5
        else: vol_score = 0.2

    # 최종 점수 계산
    score = (
        alpha_score * w_alpha +
        sharpe_score * w_sharpe +
        beta_score * w_beta +
        vol_score * w_vol
    )
    
    return min(score, 1.0)

def _get_investment_grade_kr(score: float) -> str:
    if score >= 0.8: return "A+ (매우 우수)"
    elif score >= 0.7: return "A (우수)"
    elif score >= 0.6: return "B+ (양호)"
    elif score >= 0.5: return "B (보통)"
    elif score >= 0.4: return "C (미흡)"
    else: return "D (부진)"

def _get_investment_reasons_kr(alpha: float, beta: float, sharpe: float, volatility: float) -> list:
    reasons = []
    if alpha > 0.05: reasons.append(f"우수한 알파({alpha*100:.1f}%): 시장 대비 초과 수익 발생")
    else: reasons.append(f"알파({alpha*100:.1f}%): 시장 수익률과 유사하거나 하회")
    
    if sharpe > 1.0: reasons.append(f"높은 샤프 지수({sharpe:.2f}): 위험 대비 수익 창출 능력 우수")
    elif sharpe > 0.5: reasons.append(f"양호한 샤프 지수({sharpe:.2f}): 위험 대비 수익 창출 능력 양호")
    else: reasons.append(f"낮은 샤프 지수({sharpe:.2f}): 위험 대비 수익 창출 능력 보통/미흡")

    if beta > 1.2: reasons.append(f"높은 베타({beta:.2f}): 시장보다 변동성이 큰 경향")
    elif beta < 0.8: reasons.append(f"낮은 베타({beta:.2f}): 시장보다 변동성이 낮은 경향 (방어적)")
    else: reasons.append(f"적정 베타({beta:.2f}): 시장과 유사한 수준의 변동성")

    if volatility > 0.4: reasons.append(f"높은 변동성({volatility*100:.1f}%): 주가 등락폭이 큼")
    else: reasons.append(f"안정적 변동성({volatility*100:.1f}%): 주가 등락폭이 안정적")
    
    return reasons

def _get_final_recommendation_kr(score: float, grade: str) -> str:
    if score >= 0.7: return "긍정적 투자 기회로 판단됨"
    elif score >= 0.5: return "신중하게 투자를 고려해볼 수 있음"
    elif score >= 0.3: return "투자 위험성 검토 필요"
    else: return "투자 위험성이 높아 추천하지 않음"

def _generate_analysis_summary(analysis_periods: Dict[str, Any]) -> str:
    """분석 결과를 바탕으로 요약을 생성합니다."""
    try:
        # 성공적으로 계산된 기간들만 필터링
        valid_periods = {k: v for k, v in analysis_periods.items() if "error" not in v}
        
        if not valid_periods:
            return "분석할 수 있는 데이터가 없습니다."
        
        # 변동률 추출
        changes = {k: v["change_rate"] for k, v in valid_periods.items()}
        
        # 전체적인 추세 분석
        positive_count = sum(1 for rate in changes.values() if rate > 0)
        negative_count = sum(1 for rate in changes.values() if rate < 0)
        total_periods = len(changes)
        
        if positive_count > negative_count:
            trend = "상승"
        elif negative_count > positive_count:
            trend = "하락"
        else:
            trend = "횡보"
        
        # 최고/최저 변동률
        max_change = max(changes.values())
        min_change = min(changes.values())
        max_period = max(changes.keys(), key=lambda k: changes[k])
        min_period = min(changes.keys(), key=lambda k: changes[k])
        
        summary_parts = [
            f"전체적으로 {trend} 추세를 보이고 있습니다.",
            f"최고 상승률은 {max_period} {max_change:+.1f}%입니다.",
            f"최대 하락률은 {min_period} {min_change:+.1f}%입니다."
        ]
        
        # 단기 vs 장기 분석
        short_term = ["1주", "1개월"]
        long_term = ["1년", "2년", "5년"]
        
        short_avg = sum(changes.get(p, 0) for p in short_term if p in changes) / len([p for p in short_term if p in changes])
        long_avg = sum(changes.get(p, 0) for p in long_term if p in changes) / len([p for p in long_term if p in changes])
        
        if short_avg > long_avg:
            summary_parts.append("단기적으로 장기 대비 상승세가 강합니다.")
        elif long_avg > short_avg:
            summary_parts.append("장기적으로 안정적인 상승 추세를 보입니다.")
        else:
            summary_parts.append("단기와 장기 성과가 비슷합니다.")
        
        return " ".join(summary_parts)
        
    except Exception as e:
        return f"요약 생성 중 오류가 발생했습니다: {str(e)}"


# # Fallback 도구들
FALLBACK_TOOLS = []

class StockToolkit:
    """주식 관련 도구들을 관리하는 Toolkit 클래스"""
    
    def __init__(self):
        self.tools = [
            load_all_tickers,
            get_stock_ohlcv,
            get_stock_price_change,
            get_ai_stock_recommendation
        ]
    
    def get_tools(self):
        """도구 리스트를 반환합니다."""
        return self.tools