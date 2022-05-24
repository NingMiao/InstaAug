class Scheduler:
    def __init__(self, min_value, max_value):
        self.max_value=max_value
        self.min_value=min_value
        self.max_abs_status=50 #! used to be 10 for cropping
        self.alpha=1.2#! used to be 1.1 for cropping
        self.status=0
        
        self.last_value=0
        self.small_or_large=0
        
    def step(self, value):
        action='nothing'
        
        
        if value<self.min_value:
            if self.small_or_large!=-1:
                self.small_or_large=-1
            else:
                if value>self.last_value+0.005:
                    pass
                else:
                    action='increase'
        elif value>self.max_value:
            if self.small_or_large!=1:
                self.small_or_large=1
            else:
                if value<self.last_value-0.005:
                    pass
                else:
                    action='decrease'
        else:
            self.small_or_large=0
        
        self.last_value=value
        
        if action=='increase' and self.status<self.max_abs_status:
            self.status+=1
            #print('increase {}'.format(value))
            return self.alpha
        elif action=='decrease' and self.status>-self.max_abs_status:
            self.status-=1
            #print('decrease {}'.format(value))
            return 1.0/self.alpha
        else:
            return 1.0

class Loss_Scheduler:
    def __init__(self, max_tolerance=3):
        self.current_loss=None
        self.count=0
        self.max_tolerance=max_tolerance
    def step(self, loss):
        if self.current_loss is None:
            self.current_loss=loss
            return False
        if self.current_loss > loss:
            self.current_loss=loss
            self.count=1
            return False
        else:
            self.count+=1
            if self.count>self.max_tolerance:
                self.count=0
                return True
            else:
                return False