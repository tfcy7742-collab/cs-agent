"""客服数据源：**本地模拟数据**。

⚠️ 这里的所有订单、物流、退款记录都是**手工构造的演示数据**，
不是任何真实电商系统的对接结果。README 与接口响应里都会如实标注。
这样做是为了让"查询类"能力可演示、可测试，同时绝不伪装成真实数据源。

设计要点：

* **数据集固定**：不用随机数生成，保证测试与演示可复现；
* **故意留"查不到"的情况**：库里不存在的订单号会走 not_found 分支——
  这正是"不编造"最需要被验证的路径；
* **时间基准固定**：``_TODAY`` 是一个固定日期，避免"今天变成明天后测试挂掉"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

#: 固定的"今天"（演示数据的时间基准）。用固定值而不是 date.today()，
#: 否则随着真实日期推移，"预计送达"会全部变成过去时。
TODAY = date(2026, 3, 20)


@dataclass
class OrderRecord:
    """一笔订单。"""

    order_id: str
    item: str
    category: str
    amount: float
    status: str  # paid / shipped / delivered / returning / refunded / cancelled
    paid_at: str
    shipped_at: str = ""
    delivered_at: str = ""
    carrier: str = ""
    tracking_no: str = ""
    #: 物流轨迹：[(时间, 描述), ...]（时间倒序存，展示时按时间正序）
    traces: List[Dict[str, str]] = field(default_factory=list)
    eta: str = ""
    #: 退款信息（若有）
    refund_status: str = ""
    refund_amount: float = 0.0
    refund_eta: str = ""
    returnable: bool = True
    not_returnable_reason: str = ""
    invoice_issued: bool = False
    address: str = ""
    receiver: str = ""

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "订单号": self.order_id,
            "商品": self.item,
            "类目": self.category,
            "金额": self.amount,
            "状态": STATUS_TEXT.get(self.status, self.status),
            "状态码": self.status,
            "下单时间": self.paid_at,
            "收货人": self.receiver,
            "收货地址": self.address,
        }
        if self.shipped_at:
            data["发货时间"] = self.shipped_at
        if self.delivered_at:
            data["签收时间"] = self.delivered_at
        if self.carrier:
            data["承运商"] = self.carrier
        if self.tracking_no:
            data["运单号"] = self.tracking_no
        if self.eta:
            data["预计送达"] = self.eta
        if self.refund_status:
            data["退款状态"] = self.refund_status
            data["退款金额"] = self.refund_amount
            if self.refund_eta:
                data["退款预计到账"] = self.refund_eta
        data["可退货"] = self.returnable
        if not self.returnable and self.not_returnable_reason:
            data["不可退原因"] = self.not_returnable_reason
        data["已开发票"] = self.invoice_issued
        return data


STATUS_TEXT = {
    "paid": "已付款待发货",
    "shipped": "已发货运输中",
    "delivered": "已签收",
    "returning": "退货处理中",
    "refunded": "已退款",
    "cancelled": "已取消",
}

#: 固定的模拟订单（覆盖各种状态，含一笔不可退与一笔不存在的场景）。
#: 用工厂函数返回**新的对象**，避免写操作污染进程内的初始数据。
def _build_orders() -> List[OrderRecord]:
    return [
    OrderRecord(
        order_id="SO20260101",
        item="星尘降噪蓝牙耳机 Pro",
        category="数码",
        amount=899.0,
        status="shipped",
        paid_at="2026-03-15 20:14:00",
        shipped_at="2026-03-16 09:30:00",
        carrier="顺丰速运",
        tracking_no="SF1234567890123",
        eta="2026-03-21",
        address="北京市朝阳区建国路 88 号 3 单元 1502",
        receiver="张先生",
        traces=[
            {"time": "2026-03-19 18:02", "desc": "【北京市】快件已到达 北京朝阳集散中心"},
            {"time": "2026-03-18 07:41", "desc": "【上海市】快件已离开 上海转运中心，发往北京"},
            {"time": "2026-03-16 09:30", "desc": "【上海市】顺丰速运已揽收"},
        ],
    ),
    OrderRecord(
        order_id="SO20260102",
        item="轻薄羽绒服（藏青 M）",
        category="服饰",
        amount=459.0,
        status="delivered",
        paid_at="2026-03-05 11:02:00",
        shipped_at="2026-03-05 17:20:00",
        delivered_at="2026-03-08 14:05:00",
        carrier="中通快递",
        tracking_no="ZT9988776655443",
        address="北京市朝阳区建国路 88 号 3 单元 1502",
        receiver="张先生",
        traces=[
            {"time": "2026-03-08 14:05", "desc": "【北京市】快件已签收，签收人：本人"},
            {"time": "2026-03-07 21:10", "desc": "【北京市】快件已到达 北京朝阳网点，派送中"},
            {"time": "2026-03-05 17:20", "desc": "【杭州市】中通快递已揽收"},
        ],
        # 已签收 12 天 → 超过 7 天无理由窗口，但仍可因质量问题退
        returnable=True,
        not_returnable_reason="",
    ),
    OrderRecord(
        order_id="SO20260103",
        item="定制刻字情侣对戒",
        category="定制商品",
        amount=1280.0,
        status="delivered",
        paid_at="2026-03-01 10:00:00",
        shipped_at="2026-03-02 15:00:00",
        delivered_at="2026-03-06 16:30:00",
        carrier="京东物流",
        tracking_no="JD5566778899001",
        address="上海市浦东新区世纪大道 100 号",
        receiver="李女士",
        traces=[
            {"time": "2026-03-06 16:30", "desc": "【上海市】快件已签收"},
            {"time": "2026-03-02 15:00", "desc": "【深圳市】京东物流已揽收"},
        ],
        # 定制类商品不支持无理由退货（政策里有对应条款）
        returnable=False,
        not_returnable_reason="定制类商品不支持七天无理由退货（质量问题除外）",
    ),
    OrderRecord(
        order_id="SO20260104",
        item="有机棉婴儿连体衣 2 件装",
        category="母婴",
        amount=199.0,
        status="returning",
        paid_at="2026-03-10 09:12:00",
        shipped_at="2026-03-10 18:00:00",
        delivered_at="2026-03-13 12:40:00",
        carrier="圆通速递",
        tracking_no="YT1122334455667",
        address="广州市天河区天河路 200 号",
        receiver="王女士",
        traces=[
            {"time": "2026-03-18 10:20", "desc": "【广州市】退货件已揽收"},
            {"time": "2026-03-13 12:40", "desc": "【广州市】快件已签收"},
        ],
        refund_status="退货运输中，等待仓库签收质检",
        refund_amount=199.0,
        refund_eta="仓库签收后 1-3 个工作日审核，审核通过后 1-7 个工作日到账",
    ),
    OrderRecord(
        order_id="SO20260105",
        item="轻盈跑鞋（白 42）",
        category="运动",
        amount=629.0,
        status="refunded",
        paid_at="2026-02-25 14:30:00",
        shipped_at="2026-02-25 20:10:00",
        delivered_at="2026-02-28 11:15:00",
        carrier="顺丰速运",
        tracking_no="SF7788990011223",
        address="成都市武侯区人民南路四段 12 号",
        receiver="陈先生",
        traces=[
            {"time": "2026-03-06 09:00", "desc": "【成都市】退款已完成"},
            {"time": "2026-03-02 16:20", "desc": "【成都市】退货件仓库已签收"},
        ],
        refund_status="退款成功",
        refund_amount=629.0,
        refund_eta="已于 2026-03-06 原路退回",
        ),
    ]


class OrderRepository:
    """订单数据的只读访问层。

    刻意做成"查不到就返回 None"的语义，而不是抛异常或返回空对象——
    让"没有这笔订单"成为一个**显式的、必须被上层处理**的结果。
    """

    def __init__(self, orders: Optional[List[OrderRecord]] = None) -> None:
        self._orders: Dict[str, OrderRecord] = {
            item.order_id.upper(): item for item in (orders or _build_orders())
        }

    def get(self, order_id: str) -> Optional[OrderRecord]:
        if not order_id:
            return None
        return self._orders.get(order_id.strip().upper())

    def all(self) -> List[OrderRecord]:
        return list(self._orders.values())

    def ids(self) -> List[str]:
        return sorted(self._orders)

    def update(self, order: OrderRecord) -> None:
        """写操作（P4 的"提交退货申请"会用到）。"""
        self._orders[order.order_id.upper()] = order


#: 全局单例（接口与工具共用同一份数据集）
_repository: Optional[OrderRepository] = None


def get_repository() -> OrderRepository:
    global _repository
    if _repository is None:
        _repository = OrderRepository()
    return _repository


def reset_repository() -> None:
    """测试用：丢弃改动，恢复初始数据集。

    注意要用工厂函数**重建**记录对象，而不是复用模块级的 ``_ORDERS``：
    写操作是就地修改 dataclass 的，直接传同一个列表会把改动永久留在进程里，
    导致"单独跑通过、一起跑失败"的跨用例污染（实测踩过）。
    """
    global _repository
    _repository = OrderRepository(_build_orders())


def days_between(start: str, end: date = TODAY) -> Optional[int]:
    """计算"从某天到今天"的天数（用于判断是否还在无理由退货窗口内）。"""
    if not start:
        return None
    try:
        started = date.fromisoformat(start[:10])
    except ValueError:
        return None
    return (end - started).days


def in_return_window(order: OrderRecord, window_days: int = 7) -> bool:
    """是否还在"七天无理由"窗口内（以签收时间为起点；未签收以发货时间为起点）。"""
    base = order.delivered_at or order.shipped_at or order.paid_at
    elapsed = days_between(base)
    if elapsed is None:
        return False
    return elapsed <= window_days


def next_days(days: int) -> str:
    """基于固定基准日生成日期字符串（演示用，避免依赖真实系统时间）。"""
    return (TODAY + timedelta(days=days)).isoformat()
