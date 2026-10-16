from public_lab.run import run
import pytest
def test_numeric_completion_and_truncation():
    x={"text":"Work: 600 + 600 = 1200.\nThe final answer is $1,200.00","gold":"1200","finish_reason":"stop"}
    assert run(x)=={"answer":"1200","correct":True,"truncated":False}
    assert run({**x,"finish_reason":"length"})=={"answer":None,"correct":False,"truncated":True}
    assert not run({**x,"text":"The final answer is 1,2"})['correct']
    with pytest.raises(ValueError):run({**x,"text":"x"*4097})
