
The goal of this exercise to evaluate a candidate on their ability to dive into a codebase, debug, and improve it. 

The evaluation script is a binary classification evaluation suite (with accuracy as teh metric).  

There's 3 major issues with the code: 
1) is_correct should be True/False, not 0/1 
2) Exception should be wrapped in string
3) we should use lock. 

As an extension, ask interviewee to refactor/clean up codebase.


