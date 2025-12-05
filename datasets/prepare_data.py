"""
SMS Spam Collection Dataset
Source: UCI Machine Learning Repository

This is a public dataset containing SMS messages labeled as spam or ham (not spam).
Format: label\tmessage

Download from: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

For this project, we'll create a sample dataset for demonstration.
In production, you would use the full dataset or your own data.
"""

# Sample spam and ham messages for training
SAMPLE_DATA = """ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
ham	Ok lar... Joking wif u oni...
spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
ham	U dun say so early hor... U c already then say...
ham	Nah I don't think he goes to usf, he lives around here though
spam	FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv
ham	Even my brother is not like to speak with me. They treat me like aids patent.
ham	As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune
spam	WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.
spam	Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030
ham	I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.
spam	SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info
spam	URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18
ham	I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times.
ham	I HAVE A DATE ON SUNDAY WITH WILL!!
spam	XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL
ham	Oh k...i'm watching here:)
ham	Eh u remember how 2 spell his name... Yes i did. He v naughty make until i v wet.
spam	Fine if that's the way u feel. That's the way its gota b
spam	England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/ú1.20 POBox 36504 W45WQ 16+
ham	Is that seriously how you spell his name?
ham	I'm going to try for 2 months ha ha only joking
spam	So ü pay first lar... Then when is da stock comin...
ham	Aft i finish my lunch then i go str down lor. Ard 3 smth lor. U finish ur lunch already?
spam	Ffffffffff. Alright no way I can meet up with you sooner?
ham	Just forced myself to eat a slice. I'm really not hungry tho. This sucks. Mark is getting worried. He knows I'm sick when I turn down pizza. Lol
spam	Lol your always so convincing.
ham	Did you catch the bus ? Are you frying an egg ? Did you make a tea? Are you eating your mom's left over dinner ? Do you feel my Love ?
spam	I'm back & we're packing the car now, I'll let you know if there's room
ham	Ahhh. Work. I vaguely remember that! What does it feel like? Lol
spam	Wait that's still not all that clear, were you not the one who made it seem like you were going to pay for the content?
ham	Yeah he got in at 2 and was v apologetic. n had fallen out and she was actin like we always do and he really was exhausted. He doesnt want to be accountable for me
spam	K tell me anything about you.
ham	For fear of fainting with the of all that housework you just did? Quick have a cuppa
spam	Thanks for your subscription to Ringtone UK your mobile will be charged £5/month Please confirm by replying YES or NO. If you reply NO you will not be charged
spam	REMINDER FROM O2: To get 2.50 pounds free call credit and details of great offers pls reply 2 this text with your valid name, house no and postcode
ham	Will ü b going to esplanade fr home?
spam	Also andros ice etc mustn't really b in the place yet coz it hasn't been cold enough? You neither
spam	No calls..messages..missed calls
ham	After its finished, you can bring it back. Its always here.
spam	Didn't you get hep b immunisation in nigeria.
ham	Fair enough, anything going on?
spam	Yeah hopefully, if tyler can't do it I could maybe ask around a bit
ham	U don't know how stubborn I am. I didn't even want to go to the hospital. I kept telling Mark I'm not a weak sucker. Hospitals are for weak suckers.
ham	Ğud evening my heart can v meet in coming weekend?
spam	For me the love should start with attraction.i should feel that I need her every time around me.she should be the first thing which comes in my thoughts.I would start the day and end it with her.
ham	Do you know what Mallika Sherawat did yesterday?
spam	Ok...will forget that for now. You can be my slave for tonight atleast!
ham	Fine if that's the way u feel. That's the way its gota b
spam	Plus the vodafone shares are doing quite poorly, how about the ones which you mentioned to me, if you had them with you right now
ham	Ok i thk i got it. So pick the 8th one from beginning. Is that correct?
spam	Wow! I never say anything to hurt you, I'm just not much of a talker I think.
ham	Can you please send me your account number and sort code? I've just realised I don't have it.
spam	Make money from your mobile! Up to 750 pounds per week by working for SMS chat! Info - call 08707509020 or start now and send a msg w keyword start
spam	Our brand new mobile music service is now available for you. It has free credit and will be sent to you completely FREE. Respond 'FREE' or 'STOP' on 8007 1919 16
ham	Yeh. You'll be able to see me perform though I guess that's not much consolation for you."""


def create_sample_dataset():
    """Create a sample dataset file"""
    import os

    datasets_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)

    filepath = os.path.join(datasets_dir, 'spam_sample.txt')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(SAMPLE_DATA)

    print(f"Sample dataset created at: {filepath}")
    return filepath


if __name__ == '__main__':
    create_sample_dataset()
