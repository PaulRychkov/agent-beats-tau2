import json
import logging
import os
from openai import AsyncOpenAI
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIRST_MESSAGE_SEPARATOR = "Now here are the user messages:"
TOOLS_SECTION_START = "Here's a list of tools you can use"

REASONING_GUIDELINES_AIRLINE = """
CRITICAL REASONING GUIDELINES FOR AIRLINE DOMAIN — read carefully before every response:

### 1. CABIN CLASS CHANGES AND BASIC ECONOMY RULES (VERY IMPORTANT)
The policy states: "all reservations, INCLUDING BASIC ECONOMY, can change cabin without changing the flights."
- Basic economy CAN change cabin class. NEVER refuse a cabin class change just because the reservation is basic economy.
- The "basic economy cannot be modified" rule applies ONLY to changing the flight itinerary (flight numbers/dates), NOT to cabin class.
- Cabin class must be the SAME across ALL flight segments AND ALL passengers in a reservation. You CANNOT change cabin for just one leg of a round trip, or for just one passenger. Politely refuse such requests.
- Strategy when user wants to change flight numbers/dates on basic_economy:
  a) If user ALSO wants to upgrade cabin: first update cabin class only (same flights), then update flight itinerary.
  b) If user does NOT want to change cabin (stay basic_economy): reservation cannot be modified. Inform user, offer to: cancel + rebook as a new reservation, OR first upgrade cabin then change flights.
- Do BOTH steps (cabin change + flight change) when user requests both.

### 2. ORIGIN AND DESTINATION CANNOT BE CHANGED
- The origin city/airport and destination city/airport of a reservation CANNOT be changed under any circumstances.
- If user wants a different origin or destination: inform them the change is not possible, and offer to cancel the current reservation and book a new one with the desired cities.
- Do NOT transfer to a human agent for this — handle it yourself by offering cancel+rebook.

### 3. SUPERVISOR / HUMAN TRANSFER REQUESTS
- Transfer to human ONLY when the request is truly outside your tool capabilities.
- When a user asks for a supervisor or insists they have a different membership level: use official system records, answer their original question based on those records, explain the discrepancy politely. Do NOT transfer.
- Emotional reactions, complaints, membership disputes, origin/destination changes are NOT valid transfer reasons.

### 4. CANCELLATION — WHEN TO DENY vs WHEN TO ALLOW
Cancellation is allowed ONLY if one of these is true:
  a) Booking was made within the last 24 hours
  b) Flight was cancelled by the airline
  c) It is a business class reservation
  d) User has travel insurance AND reason is health or weather
If NONE apply: DENY the cancellation with a clear explanation. Do NOT transfer to human.
A past flight (departure date already passed) CANNOT be cancelled — inform the user.
If the user has multiple reservations: check each one separately, cancel only the eligible ones, deny the rest.

### 5. PAYMENT METHODS — KEY RULES
- For FLIGHT CHANGES (update_reservation_flights): user must provide ONE gift card OR credit card. Travel certificates CANNOT be used for flight changes.
- For NEW BOOKINGS: up to 1 travel certificate + 1 credit card + up to 3 gift cards.
- Only 1 certificate per reservation (even if user has multiple certificates — use only 1 per booking).
- Gift cards are valid even with small balances (e.g. $35 is fine).
- Always check ALL payment methods before claiming payment is impossible.

### 6. PRICING CABIN CLASS CHANGES
To get the new price after a cabin change:
1. Use search_direct_flight or search_onestop_flight for the same routes/dates in the NEW cabin class.
2. Sum new prices across ALL passengers × ALL flight segments.
3. Compare total to the original amount paid.
4. New > original → user pays the difference. New < original → user gets a refund.
Do NOT use get_flight_status — it does not return prices.

### 7. FREE BAG CALCULATION
Free bags per passenger by membership and cabin:
  Regular: basic_economy=0, economy=1, business=2
  Silver:  basic_economy=1, economy=2, business=3
  Gold:    basic_economy=2, economy=3, business=4
Extra bags: $50 each. Charge only for bags ABOVE the free allowance.

### 8. PAYMENT OPTIMIZATION FOR NEW BOOKINGS
When multiple payment methods are available:
1. Use ALL gift cards first (up to 3), applying their full balances.
2. Use 1 travel certificate (max 1 per reservation).
3. Put the remaining amount on the credit card.

### 9. SEARCHING FOR CHEAPEST FLIGHTS
- Economy and Basic Economy are DIFFERENT cabin classes. "Cheapest Economy" excludes basic economy.
- Search direct flights first; if none found, search one-stop flights.
- For multi-leg reservations, search each leg separately with the correct date.

### 10. MULTI-RESERVATION TASKS
When the user mentions "all my reservations" or wants to act on multiple bookings:
- First call get_user_details to get the full list of reservation_ids for the user.
- Then retrieve EACH reservation independently using get_reservation_details.
- Evaluate each one individually (cancellation eligibility, flight duration, upgrade eligibility, etc.).
- Do not skip any reservation or make assumptions without checking.

---
Now follow the domain policy:
"""

REASONING_GUIDELINES_RETAIL = """
CRITICAL REASONING GUIDELINES FOR RETAIL DOMAIN — read carefully before every response:

### 1. AUTHENTICATION IS MANDATORY
- You MUST verify user identity at the start of EVERY conversation using find_user_id_by_email OR find_user_id_by_name_zip.
- Do this even if the user provides their user ID directly — policy requires verification first.

### 2. MODIFY/EXCHANGE/RETURN TOOL CAN ONLY BE CALLED ONCE PER ORDER
- modify_pending_order_items and exchange_delivered_order_items can each only be called ONCE per order.
- Before calling, collect ALL items the user wants to change into a single list.
- Ask the user to confirm they have provided ALL items before proceeding.
- After the call the order status changes and you cannot modify it again.

### 3. RETURN vs EXCHANGE — DIFFERENT ACTIONS
- Return: user gets a refund to original payment method or an existing gift card.
- Exchange: user swaps items for a different variant of the same product type. No change of product type allowed.
- If the user says "exchange" but describes returning items for refund, use return. Ask to clarify.

### 4. PAYMENT METHOD RULES
- Modify payment: user can only switch to a SINGLE new payment method (not the original one).
- If switching to a gift card: it must have enough balance to cover the full order total.
- Return/exchange price difference: must go to original payment method OR an existing gift card.

### 5. CANCELLATION RULES
- Can only cancel PENDING orders.
- Accepted reasons: "no longer needed" or "ordered by mistake". Other reasons are not acceptable.

### 6. MULTI-ORDER TASKS
When the user wants to act on multiple orders:
- Look up each order with get_order_details separately.
- Check status of each before acting (pending vs delivered).
- Use the correct tool: cancel for pending, return/exchange for delivered.

### 7. CALCULATING TOTALS
When asked for a total refund amount across multiple orders, use the calculate tool rather than doing mental arithmetic. Communicate the exact amount to the user.

---
Now follow the domain policy:
"""

REASONING_GUIDELINES_TELECOM = """
CRITICAL REASONING GUIDELINES FOR TELECOM DOMAIN — read carefully before every response:

### 1. IDENTIFY THE CUSTOMER FIRST
- Always look up the customer before taking any action.
- Lookup methods: phone number, customer ID, or full name + date of birth.

### 2. TECHNICAL SUPPORT — FOLLOW THE TROUBLESHOOTING WORKFLOW
- For mobile data issues, work through the diagnostic steps systematically.
- Common causes (check in order): airplane mode on, data mode off, data saver on, bad network preference, VPN interfering, data usage exceeded, roaming disabled when abroad.
- Try ALL relevant steps before transferring to human agent.
- After fixing settings, ask the user to reboot their device if needed.

### 3. ROAMING — TWO SEPARATE ACTIONS
- "Enable roaming" (enable_roaming) = grants the account permission to roam.
- "Toggle roaming" (toggle_roaming) = turns the data roaming ON on the device.
- If the user is abroad and roaming was never enabled: call enable_roaming THEN toggle_roaming.
- If roaming is already enabled but toggled off: call only toggle_roaming.

### 4. DATA REFUELING
- Maximum refuel amount is 2 GB per request.
- Confirm the price with the user before applying.

### 5. LINE SUSPENSION
- Can lift suspension ONLY if user has paid all overdue bills.
- Cannot lift suspension if the contract end date is in the past — even after payment.
- After resuming line: tell user to reboot their device.

### 6. OVERDUE BILL PAYMENT FLOW
1. Verify bill is overdue.
2. Send payment request (status becomes AWAITING PAYMENT).
3. Ask user to check_payment_request.
4. Call make_payment after user confirms.
5. Verify bill status is PAID before confirming to user.

### 7. PLAN CHANGES
- Gather available plans, let user choose.
- Calculate and confirm the new price before applying.

---
Now follow the domain policy:
"""

REASONING_GUIDELINES_DEFAULT = """
CRITICAL REASONING GUIDELINES:
- Transfer to human ONLY when the request is truly outside your capabilities.
- Always verify facts using available tools before taking action.
- If policy does not allow an action, deny the request clearly — do NOT transfer to human.
- Confirm all action details with the user before executing any database-modifying operation.

---
Now follow the domain policy:
"""


def get_reasoning_guidelines(system_text: str) -> str:
    text_lower = system_text.lower()
    if "airline agent policy" in text_lower or "airline agent" in text_lower:
        return REASONING_GUIDELINES_AIRLINE
    elif "retail agent policy" in text_lower or "retail agent" in text_lower:
        return REASONING_GUIDELINES_RETAIL
    elif "telecom agent policy" in text_lower or "telecom agent" in text_lower:
        return REASONING_GUIDELINES_TELECOM
    return REASONING_GUIDELINES_DEFAULT


class Agent:
    def __init__(self):
        self.history: list[dict] = []
        self.system_prompt: str | None = None
        self.tools: list[dict] = []
        self.pending_tool_call_id: str | None = None
        self.turn = 0
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        )
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    def _parse_first_message(self, text: str) -> tuple[str, str]:
        if FIRST_MESSAGE_SEPARATOR in text:
            system_part, user_part = text.split(FIRST_MESSAGE_SEPARATOR, 1)
            guidelines = get_reasoning_guidelines(system_part)
            augmented = guidelines + system_part.strip()
            return augmented, user_part.strip()
        guidelines = get_reasoning_guidelines(text)
        return guidelines, text

    def _extract_tools(self, system_text: str) -> list[dict]:
        """Parse OpenAI-compatible tool definitions from the system prompt text."""
        try:
            start = system_text.find('[', system_text.find(TOOLS_SECTION_START))
            if start == -1:
                return []
            decoder = json.JSONDecoder()
            tools, _ = decoder.raw_decode(system_text, start)
            return [t for t in tools if t.get('function', {}).get('name') != 'respond']
        except Exception as e:
            logger.warning("Failed to extract tools: %s", e)
            return []

    def _is_tool_result(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped.startswith(('{', '[')):
            return False
        try:
            json.loads(stripped)
            return True
        except json.JSONDecodeError:
            return False

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        self.turn += 1
        input_text = get_message_text(message)

        if self.system_prompt is None:
            system_ctx, user_content = self._parse_first_message(input_text)
            self.system_prompt = system_ctx
            self.tools = self._extract_tools(system_ctx)
            if user_content:
                self.history.append({"role": "user", "content": user_content})
            logger.info("[turn %d] INIT tools=%d | user: %s", self.turn, len(self.tools), user_content[:200])
        else:
            if self.pending_tool_call_id:
                self.history.append({
                    "role": "tool",
                    "tool_call_id": self.pending_tool_call_id,
                    "content": input_text,
                })
                self.pending_tool_call_id = None
                logger.info("[turn %d] TOOL: %s", self.turn, input_text[:200])
            else:
                self.history.append({"role": "user", "content": input_text})
                logger.info("[turn %d] USER: %s", self.turn, input_text[:200])

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )

        try:
            kwargs: dict = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    *self.history,
                ],
                "temperature": 0.0,
            }
            if self.tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"
                kwargs["parallel_tool_calls"] = False

            response = await self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                tool_call = choice.message.tool_calls[0]
                self.pending_tool_call_id = tool_call.id
                self.history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc.model_dump() for tc in choice.message.tool_calls],
                })
                reply = json.dumps({
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                })
            else:
                content = choice.message.content or ""
                self.history.append({"role": "assistant", "content": content})
                reply = json.dumps({
                    "name": "respond",
                    "arguments": {"content": content},
                })

            logger.info("[turn %d] AGENT: %s", self.turn, reply[:300])

        except Exception as e:
            logger.error("[turn %d] Error: %s", self.turn, e)
            reply = json.dumps({
                "name": "respond",
                "arguments": {"content": "I'm sorry, I encountered an error. Please try again."},
            })

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=reply))],
            name="response",
        )
