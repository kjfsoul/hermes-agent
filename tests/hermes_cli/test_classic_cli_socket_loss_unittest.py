"""Deterministic regression for classic CLI lost-socket recovery. No gateway or models."""
from __future__ import annotations
from queue import Queue
from types import SimpleNamespace
import unittest

from hermes_cli.gateway_agent import GatewayAgent, _Turn
from hermes_cli.gateway_client import GatewayClientError, _client_ping_interval
from hermes_cli.gateway_connection import DISCONNECTED

class FakeConnection:
    def __init__(self, *, pending=None, receipt=None, epoch=2):
        self.events=Queue()
        self.epoch=epoch
        self.calls=[]
        self.pending=pending or []
        self.receipt=receipt or {"status":"terminal", "result":{"final_response":"already completed","completed":True}}
    def reattach(self, session_id, *, abandon):
        self.calls.append("reattach")
        return {"pending":self.pending, "prompts": []}
    def call(self, method, **kwargs):
        self.calls.append(method)
        if method=="prompt.receipt":return self.receipt
        raise AssertionError("Only existing prompt receipt allowed: "+method)

def agent_for(conn):
    agent=GatewayAgent.__new__(GatewayAgent)
    turn=_Turn("local-abc")
    agent._turn=turn
    agent.session_id="local-abc"
    agent._interrupt_sent_at=None
    agent._epoch=1
    agent._admission=turn.admission="adm-1"
    agent.connection=conn
    agent._state=lambda: SimpleNamespace(execution_generation=None)
    agent._restore_connection_card=lambda _:None
    return agent,turn

def lost(epoch=1):
    exc=GatewayClientError(DISCONNECTED)
    exc.connection_epoch=epoch
    return exc

def finished():
    return {"method":"event","params":{"session_id":"local-abc",
        "type":"message.complete","admission_id":"adm-1",
        "payload":{"text":"new output","outcome":"completed"}}}

class TestClassicCliSocketLoss(unittest.TestCase):
    def test_loopback_avoids_client_ping_timeout_but_remote_keeps_it(self):
        for url in ("ws://127.0.0.1:8222", "ws://localhost:8222", "ws://[::1]:8222"):
            self.assertIsNone(_client_ping_interval(url), url)
        self.assertEqual(_client_ping_interval("wss://agent.example.org/ws"), 20.0)

    def test_loss_reconciles_same_admission_receipt_no_new_submit(self):
        conn=FakeConnection()
        agent,turn=agent_for(conn)
        conn.events.put(lost())
        out=agent._pump(None,{},turn)
        self.assertEqual(out,{"_receipt":conn.receipt})
        self.assertEqual(conn.calls,["reattach","prompt.receipt"])
    def test_loss_after_started_continues_original_turn_without_resubmit(self):
        conn=FakeConnection(pending=[{"admission_id":"adm-1","status":"started"}])
        agent,turn=agent_for(conn)
        conn.events.put(lost())
        conn.events.put(finished())
        out=agent._pump(None,{},turn)
        self.assertEqual(out["text"],"new output")
        self.assertEqual(conn.calls,["reattach"])
        self.assertEqual(agent._epoch,2)
    def test_replayed_stale_socket_loss_does_not_recover(self):
        conn=FakeConnection(epoch=3)
        agent,turn=agent_for(conn)
        agent._epoch=2
        conn.events.put(lost(1))
        conn.events.put(finished())
        out=agent._pump(None,{},turn)
        self.assertEqual(out["text"],"new output")
        self.assertEqual(conn.calls,[])
    def test_unrelated_exception_still_fails_without_recovery(self):
        conn=FakeConnection()
        agent,turn=agent_for(conn)
        problem=ValueError("not a socket drop")
        conn.events.put(problem)
        self.assertIs(agent._pump(None,{},turn),problem)
        self.assertEqual(conn.calls,[])
    def test_failed_receipt_is_not_fabricated_success(self):
        conn=FakeConnection(pending=[{"admission_id":"adm-1","status":"unknown"}])
        agent,turn=agent_for(conn)
        conn.events.put(lost())
        out=agent._pump(None,{},turn)
        self.assertIsInstance(out, GatewayClientError)
        self.assertIn("replay_gap",str(out))
        self.assertEqual(conn.calls,["reattach"])

if __name__=="__main__": unittest.main()
